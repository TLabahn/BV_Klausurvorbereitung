import cv2
import matplotlib.pyplot as plt
from src import bildverarbeitung as bv


def main():
    # Load image
    img_path = 'img/mandril_gray.png'
    print(f'Load image: {img_path}')
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is not None:
        print(f'Image {img_path} loaded successfully')

        # Call function here
        img_4bit = bv.rescale_greylevels(img)

        # Plot image
        plt.imshow(img_4bit, cmap='gray', vmin=0, vmax=15)
        plt.colorbar()
        plt.show()

    else:
        print(f'Bild {img_path} konnte nicht geladen werden.')


if __name__ == "__main__":
    main()
