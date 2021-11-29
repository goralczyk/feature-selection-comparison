import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as Tree

available_embedded = {"lasso": Lasso, "ridge": Ridge, "net": ElasticNet, "random_forest": RandomForestClassifier,
                      "lr": LogisticRegression, "lrcv": LogisticRegressionCV}
available_wrappers = {"rfe": RFE, "rfecv": RFECV}
available_classifiers = {"svm": SVC, "reg": LogisticRegression, "lda": LDA, "qda": QDA, "tree": Tree}
extraction_methods = ["coef_", "feature_importances_", "ranking_"]


class MisclassificationRate:
    def validate(X_pred, y_t):
        return np.sum(np.round(X_pred) != y_t)/X_pred.shape[0]


class JsonIOMixin:
    def from_json(self, model_descr):
        self.components = {}
        available_wrapable = {**available_embedded, **available_classifiers}
        for name, description in model_descr["components"].items():
            component, cmp_params = next(iter(description.items()))
            if component in available_wrappers.keys():
                wrapper = available_wrappers[component]
                wrapable_keys = cmp_params.keys() & available_wrapable.keys()
                if not wrapable_keys or len(wrapable_keys) > 1 :
                    pass # Throw warning that wrapper won't be included because of lack of classifier/too many classifiers
                else:
                    wrapable_key = wrapable_keys.pop()
                    wrapable = available_wrapable[wrapable_key](**cmp_params[wrapable_key])
                    parsed_params = cmp_params.copy()
                    parsed_params["estimator"] = wrapable
                    parsed_params.pop(wrapable_key,None)
                    self.components[name] = wrapper(**parsed_params)
            else:
                self.components[name] = available_embedded[component](**cmp_params)
        return self
    
    def to_json(self):
        def reverse_dict(x):
            return {j:i for i,j in x.items()}

        def get_arguments(component):
            parameter_values = component.__dict__.items()
            constructor_argument_names = component.__init__.__code__.co_varnames
            return {k: v for k,v in parameter_values if k in constructor_argument_names}
        rev_components = reverse_dict(available_embedded)
        rev_wrappers = reverse_dict(available_wrappers)
        rev_wrappable = reverse_dict({**available_embedded, **available_classifiers})
        output = {}
        output['components'] = {}
        components = self.components
        for name, component in components.items():
            output['components'][name] = {}
            component_key = rev_components.get(component.__class__)
            if component_key is not None:
                component_params = get_arguments(component)
                output['components'][name][component_key] = component_params
            else:
                component_key = rev_wrappers.get(component.__class__)
                if component_key is not None:
                    output['components'][name][component_key] = {}
                    component_params = get_arguments(component)
                    wrapped = component_params["estimator"]
                    wrapped_key = rev_wrappable.get(wrapped.__class__)
                    if wrapped_key is not None:    
                        output['components'][name][component_key][wrapped_key] = get_arguments(wrapped)
                        for k,v in component_params.items():
                            if k is not 'estimator':
                                output['components'][name][component_key][k] = v
                    else:
                        pass # throw unsuported wrappable error
                    
                else:
                    pass # throw unsuported component/wrapper error
        return output


class Model(JsonIOMixin):
    def __init__(self, components=[], ensemble=None, validation=None):
        self.ensemble = ensemble
        self.validation = validation
        if type(components) is dict:
            self.components = components
        elif type(components) is list:
            self.components = {"Component_{0}".format(name) : component for name, component in enumerate(components)}
    
    def add_component(self, new_component):
        self.components.append(new_component)
        
    def set_validation(self, validation):
        self.validation = validation
    
    def fit(self, X, y):
        for component in self.components.values():
            component.fit(X, y)
    
    def predict(self, X):
        return np.array([np.round(component.predict(X)) for component in self.components.values()])

    def predictWithComponentNames(self, X):
        return np.array([(component, component.predict(X)) for component in self.components.values()])

    def validate(self, X_t, y_t):
        if self.validation is not None:
            predictions = self.predict(X_t)
            return np.array([self.validation.validate(pred, y_t) for pred in predictions])
        else:
            pass # throw custom exception NoValidationSpecified

    def validateWithComponentNames(self, X_t, y_t):
        if self.validation is not None:
            predictions = self.predictWithComponentNames(X_t)
            return np.array([(comp, self.validation.validate(pred, y_t)) for (comp, pred) in predictions])
        else:
            pass # throw custom exception NoValidationSpecified

    def feature_ranking(self):
        def _get_proper_attribute(component):
            component_methods = set(dir(component))
            found_method = (component_methods & set(extraction_methods)).pop()
            return getattr(component, found_method)
        return np.array([_get_proper_attribute(component) for component in self.components.values()])

    def perform_voting(self):
        def adopt(feat, name):
            if type(name) is LogisticRegressionCV or type(name) is LogisticRegression:
                return np.apply_along_axis(np.sum, 0, abs(feat))
            elif type(name) is RFECV or type(name) is RFE:
                return np.apply_along_axis(lambda x: 1 / x, 0, feat)
            else:
                return feat

        def scale(raw):
            return raw / np.max(raw)

        ranking = self.feature_ranking()
        ranking = [adopt(feat, name) for feat, name in zip(ranking, list(self.components.values()))]
        ranking = np.array(ranking)
        ranking = abs(ranking)
        scaled = np.apply_along_axis(scale, 1, ranking)
        return np.apply_along_axis(np.mean, 0, scaled)

    def __repr__(self):
        return "Components:\n\n" + "\n\n".join([str(c) for c in self.components.items()]) \
               + "\n\nEnsemble method:\n\nweighted arithmetic mean applied on scaled results"
