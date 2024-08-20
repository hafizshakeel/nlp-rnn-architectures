import torch


# CBOW - predict center word using four context words
def predict_center_word(net, context_words, word_to_idx, idx_to_word):
    net.eval()
    context_idxs = torch.tensor([[word_to_idx[word] for word in context_words]], dtype=torch.long)
    with torch.no_grad():
        output = net(context_idxs)
    predicted_idx = str(output.argmax(dim=1).item())  # Convert predicted index to str if idx_to_word uses string keys

    return idx_to_word[predicted_idx]


# Skip-Gram model - predict center word using four context words
def predict_context_words(model, center_word, word_to_idx, idx_to_word):
    model.eval()
    center_idx = torch.tensor([word_to_idx[center_word]], dtype=torch.long)
    with torch.no_grad():
        output = model(center_idx)

    probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().tolist()
    top_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)

    # print(f"Top indices: {top_indices}")
    # print(f"idx_to_word: {idx_to_word}")

    predicted_words = []
    for idx in top_indices[:4]:  # top 4 context words
        idx = int(idx)  # Ensure index is an integer
        if idx in idx_to_word:
            predicted_words.append(idx_to_word[idx])
        else:
            print(f"Index {idx} not found in idx_to_word dictionary.")
            predicted_words.append(f"Unknown index {idx}")

    return predicted_words
