
class Registry():
    """
    Generates a mapping of strings to classes
    Useful to write configs that can use multiple classes as fields,
    like models, trainers, schedulers etc.
    """
    def __init__(self, name):
        self.name = name
        self._data_dict = dict()

    def register(self, module=None):
        
        # For existent module classes
        if module is not None:
            self._data_dict[module.__name__] = module
        
        # For customized classes (used as decorator)
        def _register(custom_class):
            self._data_dict[custom_class.__name__] = custom_class
        
        return _register
    
    def get(self, key):
        creator = self._data_dict.get(key, None)
        if creator is None:
            raise KeyError(f"Registry does not contain the key {key}")
        return creator
    
    def create(self, cfg):
        """
        Creates an object of the class defined by the 'type'
        field in a configuration dict if the class is registred

        Args:
            cfg (dict): configuration containing the data to build a class object
            registry (Registry): Registry where the class is mapped
        """
        
        if "type" not in cfg:
            raise AttributeError("'cfg' object does not have an 'type' attribute.")

        # Finds the class in registry
        args = cfg.deepcopy()
        class_name = args.pop("type")
        creator = self.get(class_name)
        
        # Creates object from class
        try:
            # By removing the type from the args, the rest of them are used to create the object
            return creator(**args)
        except:
            raise AttributeError(
                f"'cfg' object contains the wrong arguments to create an object of {class_name}"
            )