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
        # new_img = bv.set_contrast(img, 10)

        # Plot image
        # plt.imshow(new_img, cmap='gray', vmin=0, vmax=255)
        # plt.colorbar()
        # plt.show()

        # values = bv.get_hist(img)

        # Plot
        # plt.hist(values, bins=256, range=(0, 256), color='gray')
        # plt.title("Histogram of Greyscale Values")
        # plt.xlabel("Grey Value")
        # plt.ylabel("Pixel Count")
        # plt.grid(True)
        # plt.show()

        bv.aufgabe_3()

    else:
        print(f'Bild {img_path} konnte nicht geladen werden.')


if __name__ == "__main__":
    main()
