from cobra.io import load_json_model


def compare_models(model1, model2):

    model1_reactions = [rxn.id for rxn in model1.reactions]
    model2_reactions = [rxn.id for rxn in model2.reactions]
    shared_reactions = set(model1_reactions).intersection(set(model2_reactions))
    unique_to_model1 = set(model1_reactions) - set(model2_reactions)
    unique_to_model2 = set(model2_reactions) - set(model1_reactions)

    print(f"Model 1 has {len(model1_reactions)} reactions.")
    print(f"Model 2 has {len(model2_reactions)} reactions.")
    print(f"Shared reactions: {len(shared_reactions)}")
    print(f"Reactions unique to Model 1: {len(unique_to_model1)}")
    print(f"Reactions unique to Model 2: {len(unique_to_model2)}")


if __name__ == "__main__":
    model1 = load_json_model("path/to/first/model.json")
    model2 = load_json_model("path/to/second/model.json")
    compare_models(model1, model2)
