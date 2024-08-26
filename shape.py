import torch

new_data_path = "Results/stroke_pt/stroke.pt"

new_data_tensor = torch.load(new_data_path)


print(new_data_tensor[0][0])


# arr = []

# for i in range(1000):
#     arr.append(new_data_tensor)
# #
# # Assuming arr is a list of tensors
# arr_tensor = torch.stack(arr)  # Convert list of tensors to a single tensor
# torch.save(arr_tensor, 'test_dataset/arr_tensor.pt')  # Save the tensor
# print('saved!')
