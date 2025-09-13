import xml.etree.ElementTree as ET
import tempfile
import os
from brax import envs

def modify_gravity_in_xml(xml_file_path, gravity_multiplier, save_path=None):
    """Load XML file, modify gravity, and optionally save to separate file"""
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Find or create the option tag
    option_tag = root.find('option')
    if option_tag is None:
        # Create option tag if it doesn't exist
        option_tag = ET.SubElement(root, 'option')
        option_tag.set('timestep', '0.01')
        option_tag.set('iterations', '4')
    
    # Set gravity (default is [0, 0, -9.81])
    default_gravity = [0.0, 0.0, -9.81]
    new_gravity = [g * gravity_multiplier for g in default_gravity]
    gravity_str = f"{new_gravity[0]} {new_gravity[1]} {new_gravity[2]}"
    option_tag.set('gravity', gravity_str)
    
    # Save to separate file if path provided
    if save_path:
        tree.write(save_path, encoding='unicode', xml_declaration=True)
        print(f"Modified XML saved to: {save_path}")
    
    # Convert back to string
    return ET.tostring(root, encoding='unicode')

def create_temp_xml_file(xml_content):
    """Create a temporary XML file with the modified content"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
    temp_file.write(xml_content)
    temp_file.close()
    return temp_file.name

def create_gravity_modified_xml(original_xml_path, gravity_multiplier, output_dir="modified_envs"):
    """Create a permanent file with modified gravity"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with gravity multiplier
    base_name = os.path.splitext(os.path.basename(original_xml_path))[0]
    gravity_str = f"{gravity_multiplier:.2f}".replace('.', '_')
    output_filename = f"{base_name}_gravity_{gravity_str}.xml"
    output_path = os.path.join(output_dir, output_filename)
    
    # Modify and save XML
    modify_gravity_in_xml(original_xml_path, gravity_multiplier, save_path=output_path)
    
    return output_path

def recreate_environment_with_gravity(env_name,params, gravity_multiplier, save_file=True):
    """Recreate environment with modified gravity"""
    # Get the original XML file path
    xml_file_path = original_env.robot.xml_file
    
    if save_file:
        # Create permanent file with modified gravity
        modified_xml_path = create_gravity_modified_xml(xml_file_path, gravity_multiplier)
        
        # Recreate environment with modified XML
        new_env = envs.create(env_name=env_name,params=params, xml_file=modified_xml_path)
        
        return new_env, modified_xml_path
    else:
        # Use temporary file (original behavior)
        modified_xml_content = modify_gravity_in_xml(xml_file_path, gravity_multiplier)
        temp_xml_path = create_temp_xml_file(modified_xml_content)
        new_env = envs.create(env_name=env_name, params=params, xml_file=temp_xml_path)
        os.unlink(temp_xml_path)
        return new_env, None
    
    
def modify_gravity_directly(env_state, gravity_multiplier):
    """Modify gravity directly in the environment state"""
    # Default gravity is [0, 0, -9.81]
    default_gravity = jnp.array([0.0, 0.0, -9.81])
    new_gravity = default_gravity * gravity_multiplier
    
    # Modify the system's gravity property
    if hasattr(env_state, 'pipeline_state'):
        new_system = env_state.pipeline_state.system.replace(gravity=new_gravity)
        new_pipeline_state = env_state.pipeline_state.replace(system=new_system)
        return env_state.replace(pipeline_state=new_pipeline_state)
    
    return env_state