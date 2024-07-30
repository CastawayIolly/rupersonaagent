import numpy as np
from lime.lime_text import LimeTextExplainer
from matplotlib import pyplot as plt


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# define prediction function
def predict_probs(texts, model):
    predictions = model.predict(texts)
    x = np.array(list(predictions)[1])
    return np.apply_along_axis(softmax, 1, x)


def get_lime_explanaition(class_names, text, predict_probs, num_features):
    explainer = LimeTextExplainer(class_names=class_names)
    return explainer.explain_instance(text, predict_probs, num_features=num_features)


def vizualize_explanaition(explanation):
    explanation.as_pyplot_figure()
    plt.show()
