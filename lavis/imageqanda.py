import argparse
import os
from PIL import Image
import torch


from lavis.models import load_model_and_preprocess


def get_caption(raw_image, device):
    """
    Generates a caption for an image using a pre-trained model.

    Args:
        raw_image (torch.Tensor): The input image as a tensor.
        device (str): The device to run the model on.

    Returns:
        str: The generated caption for the input image.
    """
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return model.generate({"image": image})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Input image file to ask questions about",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise ValueError(f"Invalid image path: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_image = Image.open(args.image).convert("RGB")
    raw_image.show()

    caption = get_caption(raw_image, device)
    print(f"\n\nThe caption for this image is: {caption}\n\n")

    # Setup for image Q&A in a loop.
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_vqa", model_type="vqav2", is_eval=True, device=device
    )
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    while True:
        user_input = input("\n\nEnter your question about the image:")
        if user_input == "":
            print("Bye!")
            exit(0)

        question = txt_processors["eval"](user_input)
        answers = model.predict_answers(
            samples={"image": image, "text_input": question},
            inference_method="generate",
        )
        print(answers)
