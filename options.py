import argparse

parser = argparse.ArgumentParser(description="training argument values")


def add_training_parser(parser):
    parser.add_argument("-device", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-check_interval", type=int, default=1)
    parser.add_argument("-k_fold", type=str, default=10)
    parser.add_argument("-prompt_positive_num", type=int, default=5)
    parser.add_argument("-prompt_negative_num", type=int, default=5)
    parser.add_argument("-model_type", type=str, default="vit_h")
    parser.add_argument("-is_local", type=bool, default=False)
    parser.add_argument("-remark", type=str, default="OCTA-500")



def add_cell_parser(parser):
    parser.add_argument("-metrics", type=list, default=["Dice", "Hausdorff", "Precision", "Recall", "Specificity", "ASD"])


def add_vessel_2d_parser(parser):
    parser.add_argument("-fov", type=str, default="3M")  # Cardio
    parser.add_argument(
        "-label_type", type=str, default="LargeVessel"
    )  # "LargeVessel", "FAZ", "Capillary", "Artery", "Vein"
    parser.add_argument("-metrics", type=list, default=["Dice", "Hausdorff", "Precision", "Recall", "Specificity", "ASD"])
