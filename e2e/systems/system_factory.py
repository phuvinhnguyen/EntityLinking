"""
System factory for creating and managing different entity linking systems
"""
import sys
import os
import importlib
import inspect
from typing import Dict, Type, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from systems.base_system import BaseSystem

class SystemFactory:
    """Factory for creating entity linking systems with auto-detection"""
    
    _systems: Dict[str, Type[BaseSystem]] = {}
    _initialized = False
    
    @classmethod
    def _auto_discover_systems(cls):
        """Auto-discover all system classes in the systems directory"""
        if cls._initialized:
            return
        
        systems_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Get all Python files in the systems directory
        for filename in os.listdir(systems_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]  # Remove .py extension
                
                try:
                    # Import the module
                    module = importlib.import_module(f'systems.{module_name}')
                    
                    # Find all classes that inherit from BaseSystem
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (obj != BaseSystem and 
                            issubclass(obj, BaseSystem) and 
                            obj.__module__ == module.__name__):
                            
                            # Use class name as system name (lowercase)
                            system_name = name.lower().replace('system', '')
                            cls._systems[system_name] = obj
                            print(f"Auto-discovered system: {system_name} -> {name}")
                
                except Exception as e:
                    print(f"Error importing {module_name}: {e}")
        
        cls._initialized = True
    
    @classmethod
    def create_system(cls, system_name: str, config: Config = None) -> BaseSystem:
        """
        Create a system instance by name
        
        Args:
            system_name: Name of the system to create
            config: Configuration object
            
        Returns:
            BaseSystem: Instance of the requested system
            
        Raises:
            ValueError: If system name is not recognized
        """
        # Auto-discover systems if not already done
        cls._auto_discover_systems()
        
        system_name = system_name.lower()
        
        if system_name not in cls._systems:
            available = ', '.join(cls._systems.keys())
            raise ValueError(f"Unknown system '{system_name}'. Available systems: {available}")
        
        system_class = cls._systems[system_name]
        return system_class(config)
    
    @classmethod
    def get_available_systems(cls) -> list:
        """Get list of available system names"""
        cls._auto_discover_systems()
        return list(cls._systems.keys())
    
    @classmethod
    def register_system(cls, name: str, system_class: Type[BaseSystem]):
        """
        Register a new system class
        
        Args:
            name: Name to register the system under
            system_class: System class that inherits from BaseSystem
        """
        if not issubclass(system_class, BaseSystem):
            raise ValueError(f"System class must inherit from BaseSystem")
        
        cls._systems[name.lower()] = system_class
    
    @classmethod
    def get_system_info(cls, system_name: str) -> Dict[str, Any]:
        """
        Get information about a system
        
        Args:
            system_name: Name of the system
            
        Returns:
            Dict containing system information
        """
        cls._auto_discover_systems()
        system_name = system_name.lower()
        
        if system_name not in cls._systems:
            available = ', '.join(cls._systems.keys())
            raise ValueError(f"Unknown system '{system_name}'. Available systems: {available}")
        
        system_class = cls._systems[system_name]
        
        return {
            'name': system_name,
            'class': system_class.__name__,
            'module': system_class.__module__,
            'description': system_class.__doc__ or "No description available"
        }

def test_system_factory():
    """Test the system factory"""
    print("Testing System Factory...")
    
    # Test getting available systems
    available = SystemFactory.get_available_systems()
    print(f"Available systems: {available}")
    
    # Test creating systems
    config = Config()
    
    for system_name in available:
        print(f"\nTesting {system_name} system:")
        try:
            system = SystemFactory.create_system(system_name, config)
            print(f"  Created: {system.__class__.__name__}")
            
            # Test system info
            info = SystemFactory.get_system_info(system_name)
            print(f"  Info: {info}")
            
        except Exception as e:
            print(f"  Error creating {system_name}: {e}")
    
    # Test unknown system
    try:
        SystemFactory.create_system("unknown")
    except ValueError as e:
        print(f"\nExpected error for unknown system: {e}")
    
    print("\nSystem factory test completed!")

if __name__ == "__main__":
    test_system_factory()
