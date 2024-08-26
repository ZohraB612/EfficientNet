import torch

def modify_and_duplicate_tensor(source_path, output_path, output_path_val, copies=1000, val_copies=2):
    # Load the original tensor
    original_tensor = torch.load(source_path)

    # Check if padding is needed
    target_columns = 6
    current_columns = original_tensor.shape[1]
    if current_columns < target_columns:
        # Calculate the number of columns to add
        columns_to_add = target_columns - current_columns
        # Create a zero tensor for padding
        padding = torch.zeros((original_tensor.shape[0], columns_to_add), dtype=original_tensor.dtype)
        # Concatenate the original tensor with the padding
        modified_tensor = torch.cat((original_tensor, padding), dim=1)
    else:
        modified_tensor = original_tensor

    # Print new shape of the modified tensor
    print(f"Modified tensor shape: {modified_tensor.shape}")

    # Create an array of copies of the modified tensor
    arr = [modified_tensor for _ in range(copies)]

    # Convert list of tensors to a single tensor
    arr_tensor = torch.stack(arr)
    print(f"New tensor length: {len(arr)}")
    print(f"New tensor shape: {arr_tensor[0].shape}")

    # Save the main tensor
    torch.save(arr_tensor, output_path)
    print(f"Tensor with duplicated entries saved to {output_path}")

    # Additional functionality to create and save a smaller array of 2 tensors
    arr_val = [modified_tensor for _ in range(val_copies)]
    arr_tensor_val = torch.stack(arr_val)
    torch.save(arr_tensor_val, output_path_val)
    print(f"Tensor with 2 entries saved to {output_path_val}")

# Example usage
source_path = 'test_dataset/output_model.pt'
output_path = 'test_dataset/arr_tensor.pt'
output_path_val = 'test_dataset/arr_tensor_val.pt'

modify_and_duplicate_tensor(source_path, output_path, output_path_val)
