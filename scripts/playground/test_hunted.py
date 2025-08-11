#!/usr/bin/env python3
"""Test script for the hunted environment with HTML visualization."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jp
import brax
from brax.io import html
import tempfile
import webbrowser
import jax.numpy as jnp

# Import the hunted environment directly from the modified file
sys.path.append('.venv/lib/python3.12/site-packages')
try:
    from brax.envs.hunted import Hunted
except ImportError:
    # Fallback: try to import from the modified version
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "hunted", 
        ".venv/lib/python3.12/site-packages/brax/envs/hunted.py"
    )
    hunted_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hunted_module)
    Hunted = hunted_module.Hunted

def test_hunted_environment():
    """Test the hunted environment with random actions and generate HTML."""
    print("Testing hunted environment...")
    
    try:
        # Create environment
        env = Hunted()
        print(f"Environment created successfully")
        print(f"Action space: {env.action_size}")
        print(f"Observation space: {env.observation_size}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Reset environment
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    print(f"Environment reset successfully")
    print(f"Initial observation shape: {state.obs.shape}")
    
    # Run episode with random actions
    states = [state]
    actions = []
    rewards = []
    
    jit_step = jax.jit(env.step)
    
    episode_length = 1000
    print(f"Running episode for {episode_length} steps...")
    
    for step in range(episode_length):
        # Generate random action
        rng, action_rng = jax.random.split(rng)
        action = jax.random.uniform(action_rng, (env.action_size,), minval=-1.0, maxval=1.0)
        action = jnp.zeros_like(action)
        
        # Step environment
        state = jit_step(state, action)
        
        # Store results
        states.append(state)
        actions.append(action)
        rewards.append(state.reward)
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step}: Reward = {state.reward:.3f}, Done = {state.done}")
        
        # Check if episode is done
        if state.done:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Episode completed. Total steps: {len(states)-1}")
    print(f"Average reward: {jp.mean(jp.array(rewards)):.3f}")
    print(f"Final reward: {rewards[-1]:.3f}")
    
    # Generate HTML visualization
    print("Generating HTML visualization...")
    
    try:
        # Convert states to trajectory format
        trajectory = []
        for i, state in enumerate(states):
            # Only include pipeline_state for HTML rendering
            trajectory.append(state.pipeline_state)
        
        # Create HTML file
        html_string = html.render(env.sys, trajectory)
    except Exception as e:
        print(f"Error generating HTML: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Save to file
    output_file = "hunted_test_visualization.html"
    with open(output_file, 'w') as f:
        f.write(html_string)
    
    print(f"HTML visualization saved to: {output_file}")
    print(f"File size: {os.path.getsize(output_file)} bytes")
    
    # Try to open in browser
    try:
        webbrowser.open(f'file://{os.path.abspath(output_file)}')
        print("Opened visualization in browser")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please open {output_file} manually in your browser")
    
    return output_file

def test_hunted_metrics():
    """Test and print detailed metrics from the hunted environment."""
    print("\n" + "="*50)
    print("DETAILED METRICS TEST")
    print("="*50)
    
    env = Hunted()
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    
    print(f"Initial metrics:")
    for key, value in state.metrics.items():
        print(f"  {key}: {value}")
    
    # Run a few steps and show metrics
    for step in range(50):
        rng, action_rng = jax.random.split(rng)
        action = jax.random.uniform(action_rng, (env.action_size,), minval=-1.0, maxval=1.0)
        action = jnp.zeros_like(action)
        state = env.step(state, action)
        
        print(f"\nStep {step+1}:")
        print(f"  Reward: {state.reward:.3f}")
        print(f"  Done: {state.done}")
        print(f"  Ant position: {state.pipeline_state.x.pos[0]}")
        # Use the correct hunter position from metrics
        print(f"  Hunter position: [{state.metrics['hunter_x']:.3f}, {state.metrics['hunter_y']:.3f}, 0.2]")
        
        # Show key metrics
        key_metrics = ['hunter_distance', 'reward_escape', 'reward_caught', 'x_position', 'y_position']
        for metric in key_metrics:
            if metric in state.metrics:
                print(f"  {metric}: {state.metrics[metric]:.3f}")
        
        # Check if episode is done
        if state.done:
            print(f"Episode ended at step {step+1}")
            break

if __name__ == "__main__":
    print("Starting hunted environment test...")
    
    # Test basic functionality
    output_file = test_hunted_environment()
    
    # Test detailed metrics
    test_hunted_metrics()
    
    print(f"\nTest completed successfully!")
    print(f"Visualization saved to: {output_file}")
