class generic_callback(object):
    def __init__(self, func_object={}, value_object={}, callback_function=None):
        self.func_object = func_object
        self.value_object = value_object
        self.callback_function = callback_function
    def __call__(self):
        parameters = {k:v() for k, v in self.func_object.items()}
        parameters.update(self.value_object)
        if(self.callback_function is None):
            print("No callback function")
            return
        return self.callback_function(**parameters)
