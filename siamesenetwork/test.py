import torch


def fill_the_blank(camembert):
    # auto generate the masked word
    masked_line = 'Le camembert est <mask> :)'
    camembert.fill_mask(masked_line, topk=3)


def extract_feature(camembert):
    # Extract the last layer's features
    line = "J'aime le camembert !"
    tokens = camembert.encode(line)
    last_layer_features = camembert.extract_features(tokens)
    assert last_layer_features.size() == torch.Size([1, 10, 768])

    # Extract all layer's features (layer 0 is the embedding layer)
    all_layers = camembert.extract_features(tokens, return_all_hiddens=True)
    assert len(all_layers) == 13
    assert torch.all(all_layers[-1] == last_layer_features)


def main():
    camembert = torch.hub.load('pytorch/fairseq', 'camembert')
    camembert.eval()


if __name__ == "__main__":
    main()
