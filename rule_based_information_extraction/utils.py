from sklearn.metrics import accuracy_score, f1_score


def assign_lexical_items(full_sentence, item_dialog, item_personadescription):
    # get items that are in person description and in dialogs
    item = [i for i, j in zip(item_dialog, item_personadescription) if i == j]
    assigned_item = []
    item = list(set(item))

    for x in full_sentence:
        k = [x for w in item if w in x]
        assigned_item.append(k)

    item_tag = []
    for item_ in assigned_item:
        if len(item_) == 0:
            item_tag.append("no")
        else:
            item_tag.append("yes")
    return item_tag


def count_metrics(actual, pred):
    f1 = f1_score(actual, pred, average="weighted")
    ac = accuracy_score(actual, pred)
    return [f1, ac]


def combine_features(feature1, feauture2):
    example = []
    for i in feature1:
        for j in feauture2:
            x = "no" if (i == "no" and j == "no") else "yes"
        example.append(x)
