from functools import partial
import jax.numpy as jnp
import jax.random as jr
import jax
from jax.experimental.shard_map import shard_map as shmap
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import evosax as ex
from qdax.core.map_elites import EmitterState, Emitter, MapElitesRepertoire, MAPElites as BaseME
from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter as CMAOptEmitterBase
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.types import Centroid, Metrics, Genotype, Fitness, Descriptor, ExtraScores, RNGKey
from typing import Any, Callable, Dict, Optional, Tuple, TypeAlias, NamedTuple
from jaxtyping import Array, Float, PyTree
from src.tasks.base import QDTask

SCORING_RESULTS = Tuple[Fitness, Descriptor, ExtraScores]
class TrainState(NamedTuple):
    repertoire: MapElitesRepertoire
    emitter_state: EmitterState

def _dummy_scoring_fn(_: Genotype, k: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    return jnp.empty((1,)), jnp.empty((1,)), {'dummy': jnp.empty((1,))}, k

class MAPElites(BaseME):
    """
    Wrapper around qdax's MAP-Elites implementation.

    The qdax implementation does not use the ask-tell interface, which means that scoring functions
    cannot be easily customized. This is necessary if we are applying developmental models to the
    genotypes in the repertoire. However, this is easy to fix: we use the '_emit' function from the
    emitter to implement the ask and the 'add' and 'update' functions from the repertoire and the
    emitter, respectively, for the tell. we override the '__init__' to pass a dummy scoring function
    and delegate the task of providing the fitness values to a different component. Finally, we must
    also overwrite the 'init' function (NOTE this is the state init, not the module init) to pass
    the scores directly since the module does not have access to the scoring function.
    """
    def __init__(self,
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics]
    ) -> None:
        super().__init__(_dummy_scoring_fn, emitter, metrics_function)

    def init( #type:ignore
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        fitness_and_metrics: SCORING_RESULTS,
        key: RNGKey,
    ) -> TrainState:
        fitnesses, descriptors, extra_scores = fitness_and_metrics

        # init the repertoire
        repertoire = MapElitesRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )

        # get initial state of the emitter
        emitter_state, _ = self._emitter.init(
            init_genotypes=init_genotypes, random_key=key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return TrainState(repertoire=repertoire, emitter_state=emitter_state)


    def ask(self, train_state: TrainState, key: jax.Array) -> Genotype:
        return self._emitter.emit(train_state.repertoire, train_state.emitter_state, key)[0]


    def tell(
        self,
        genotypes: Float[Array, "..."],
        scores: SCORING_RESULTS,
        train_state: TrainState,
    ) -> TrainState:
        fitnesses, descriptors, extra_scores = scores

        # update map
        repertoire = train_state.repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=train_state.emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return TrainState(repertoire=repertoire, emitter_state=emitter_state)



#---------------------------------- Emitters ---------------------------------------

class CMAOptEmitter(CMAOptEmitterBase):
    """
    Emitter used to implement CMA-based Map-Elites (i.e. CMA-ME).
    """
    def __init__(self,
        batch_size: int,
        genotype_dim: int,
        sigma_g: float,
        min_count: Optional[int] = None,
        max_count: Optional[float] = None,
        *,
        centroid_kwargs: Dict[str, Any],
    ):
        # We must create dummy centroids to initialize the qdax emitter. Most parameters do not need
        # to match the ones actually used by a QD algorithm since these centroids are only used to
        # access their number.
        dummy_centroids, _ = compute_cvt_centroids(**centroid_kwargs)  # type: ignore
        super().__init__(batch_size, genotype_dim, dummy_centroids, sigma_g, min_count, max_count)


#-------------------------------- Scoring Function ---------------------------------

SCORING_FUNCTION = Callable[[Dict[str, Array], MapElitesRepertoire], float]


def coverage_only(qd_metrics, repertoire):
    coverage = qd_metrics['coverage']

    if len(coverage.shape):
        coverage = coverage[-1]

    return coverage / 100


def qd_score_x_coverage(qd_metrics, repertoire):
    qd_score = qd_metrics['qd_score']
    coverage = qd_metrics['coverage']

    if len(qd_score.shape):
        qd_score = qd_score[-1]
        coverage = coverage[-1]

    # coverage is a percetange, normalize it to (0, 1)
    return qd_score * coverage / 100



#-------------------------------- Trainer ---------------------------------


from src.training.base import BaseTrainer
from src.training.logging import Logger


class QDTrainer(BaseTrainer):
    
    """
    """
    #-------------------------------------------------------------------
    popsize: int
    strategy: MAPElites
    params_shaper: ex.ParameterReshaper
    task: QDTask
    n_devices: int
    centroid_initializer: Callable
    #-------------------------------------------------------------------

    def __init__(
        self, 
        strategy: MAPElites, 
        popsize: int,
        train_steps: int, 
        task: QDTask,
        params_shaper: ex.ParameterReshaper,
        logger: Optional[Logger]=None,
        progress_bar: Optional[bool]=True,
        n_devices: int=1,
        centroid_initializer: Optional[Callable]=None,
        n_centroids: Optional[int]=None,
        n_init_cvt_samples: int=256,
        minval: Float=-1.,
        maxval: Float=1.):
        
        super().__init__(train_steps=train_steps, 
                         logger=logger, 
                         progress_bar=progress_bar)
        self.popsize = popsize
        self.strategy = strategy
        self.task = task
        self.params_shaper = params_shaper
        self.n_devices = n_devices
        if centroid_initializer is not None:
            self.centroid_initializer = centroid_initializer
        else:
            assert n_centroids is not None, "must provide info for centroids"
            init_fn = partial(compute_cvt_centroids, 
                              num_descriptors=task.n_descriptors,
                              num_init_cvt_samples=n_init_cvt_samples,
                              num_centroids=n_centroids,
                              minval=minval,
                              maxval=maxval)
            self.centroid_initializer = init_fn
        
    #-------------------------------------------------------------------

    def initialize(self, key: jr.PRNGKeyArray) -> TrainState:
        
        cent_key, gen_key, eval_key, init_key = jr.split(key, 4)
        centroids = self.centroid_initializer(cent_key)
        init_genotypes = jr.normal(gen_key, (self.popsize, self.params_shaper.total_params))
        fit_desc_data = self.eval(init_genotypes, eval_key)
        train_state = self.strategy.init(init_genotypes, centroids, fit_desc_data, init_key)
        return train_state

    #-------------------------------------------------------------------

    def train_step(self, state: TrainState, key: jax.Array, data: Optional[PyTree]=None) -> Tuple[TrainState, Any]:

        ask_key, eval_key = jr.split(key)
        ask_key, eval_key = jr.split(key, 2)
        x = self.strategy_ask(state, ask_key)
        fitness, descriptors, eval_data = self.eval(x, eval_key, data)
        train_state = self.strategy_tell(state, x, fitness, descriptors, eval_data)
        return train_state, {"fitness": fitness, "data": eval_data, "descriptors": descriptors}

    #-------------------------------------------------------------------

    def eval(self, *args, **kwargs):
        
        if self.n_devices == 1:
            return self._eval(*args, **kwargs)
        else:
            return self._eval_shmap(*args, **kwargs)

    #-------------------------------------------------------------------

    def _eval(self, x: jax.Array, key: jax.Array, data: Optional[PyTree]=None)->Tuple[Float, Float, PyTree]:
        
        params = self.params_shaper.reshape(x)
        _eval = jax.vmap(self.task, in_axes=(0, None, None))
        return _eval(params, key, data)

    #-------------------------------------------------------------------

    def _eval_shmap(self, x: jax.Array, key: jax.Array, data: Optional[PyTree]=None)->Tuple[Float, Float, PyTree]:
        
        devices = mesh_utils.create_device_mesh((self.n_devices,))
        device_mesh = Mesh(devices, axis_names=("p"))

        params = self.params_shaper.reshape_single(x)
        _eval = lambda x, k: self.task(params, k, data)
        batch_eval = jax.vmap(_eval, in_axes=(0,None))
        sheval = shmap(batch_eval, 
                       mesh=device_mesh, 
                       in_specs=(P("p",), P()),
                       out_specs=(P("p"), P("p")),
                       check_rep=False)

        return sheval(x, key)

    #-------------------------------------------------------------------

    def strategy_tell(self, train_state: TrainState, x: jax.Array, 
                      fitness: jax.Array, descriptors: jax.Array, extra_scores: ExtraScores)->TrainState:
        scores = (fitness, descriptors, extra_scores)
        return self.strategy.tell(x, scores, train_state)

    #-------------------------------------------------------------------

    def strategy_ask(self, train_state: TrainState, key: jax.Array)->PyTree:
        return self.strategy.ask(train_state, key)








