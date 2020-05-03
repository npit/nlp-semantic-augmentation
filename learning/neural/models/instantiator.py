"""Instantiator for neural models"""
from learning.neural.models.bert import Bert
from learning.neural.models.mlp import MLPModel
from utils import error
from learning.neural.dnn import SupervisedDNN, UnsupervisedDNN
from learning.neural.languagemodel.language_model import SupervisedNeuralLanguageModel
from learning.neural.languagemodel.huggingface_transformer_language_model import HuggingfaceTransformerLanguageModel

available_models = [Bert, MLPModel]
available_wrappers = [SupervisedDNN, UnsupervisedDNN, HuggingfaceTransformerLanguageModel]

def get_neural_model_class(name):
    """Retrieve DNN architecture"""
    for candidate in available_models:
        if candidate.name == name:
            return candidate
    error(f"Undefined neural model: {name}")

def get_neural_wrapper_class(neural_model_name):
    """Retrieve the wrapper class of a neural model"""
    model_class = get_neural_model_class(neural_model_name)
    wrapper_name = model_class.wrapper_name
    try:
        wrapper_names = [w.name for w in available_wrappers]
        return available_wrappers[wrapper_names.index(wrapper_name)]
    except ValueError:
        error(f"Undefined neural wrapper class: {wrapper_name} for neural model {neural_model_name}")

def get_neural_model_wrapper(neural_model):
    """Retrieve wrapper class of DNN architecture"""
    neural_model = get_neural_model_class(neural_model)
    return neural_model.wrapper_class