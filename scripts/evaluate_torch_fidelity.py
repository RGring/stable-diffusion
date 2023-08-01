import torch_fidelity

def main():
    path_to_real_samples = "/home/ronja/data/generated/RumexLeaves/real_resized_images/256/val"
    path_to_generated_samples = "/home/ronja/data/generated/RumexLeaves/2023-06-09T11-09-41_rumexleaves-ldm-vq-4_pretr1_2/epoch=000099/images"
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=path_to_real_samples, 
        input2=path_to_generated_samples, 
        cuda=True, 
        isc=True, 
        fid=True, 
        kid=False, 
        prc=False, 
        verbose=False,
    )
    print(metrics_dict)

if __name__ == "__main__":
    main()