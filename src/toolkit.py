import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def select_points_on_image(pil_image):
    """
    Displays an image from a PIL Image object and allows the user to interactively select two types of points.

    Args:
        pil_image (PIL.Image.Image): The image to display.

    Returns:
        tuple: A tuple containing two lists of points (points_type1, points_type2).
    """

    try:
        # Convert the PIL image to a NumPy array
        img_np = np.array(pil_image)

        # Ensure the image is in a format that OpenCV can use (BGR)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3: # Standard RGB
            img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 4: # RGBA
            img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        elif len(img_np.shape) == 2: # Grayscale
            img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError("Unsupported image format")


        points_type1 = []
        points_type2 = []
        current_mode = 1  # 1 for type 1, 2 for type 2

        fig, ax = plt.subplots(figsize=(10, 8))
        # Display the original image (in RGB)
        ax.imshow(img_np)
        plt.axis('off')

        def update_title():
            ax.set_title(f'Mode: Points Type {current_mode} (m: switch, d: delete, close window to quit)')

        update_title()

        def redraw_points():
            # Create a fresh copy of the original BGR image to draw on
            img_with_points_bgr = img_np_bgr.copy()

            # Draw points from the first list in red
            for point in points_type1:
                cv2.circle(img_with_points_bgr, tuple(point), radius=15, color=(0, 0, 255), thickness=-1)

            # Draw points from the second list in blue
            for point in points_type2:
                cv2.circle(img_with_points_bgr, tuple(point), radius=15, color=(255, 0, 0), thickness=-1)

            # Convert back to RGB for Matplotlib and update the display
            img_with_points_rgb = cv2.cvtColor(img_with_points_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(img_with_points_rgb)
            fig.canvas.draw_idle()

        def onclick(event):
            if event.inaxes != ax:
                return

            ix, iy = int(event.xdata), int(event.ydata)
            print(f'Point selected at: (x={ix}, y={iy}) in mode {current_mode}')

            if current_mode == 1:
                points_type1.append([ix, iy])
            else:
                points_type2.append([ix, iy])

            redraw_points()

        def onkey(event):
            nonlocal current_mode
            if event.key == 'm':
                current_mode = 2 if current_mode == 1 else 1
                print(f"Switched to mode {current_mode}")
                update_title()
                fig.canvas.draw_idle()
            elif event.key == 'd':
                if current_mode == 1 and points_type1:
                    removed_point = points_type1.pop()
                    print(f"Removed point {removed_point} from Points Type 1")
                    redraw_points()
                elif current_mode == 2 and points_type2:
                    removed_point = points_type2.pop()
                    print(f"Removed point {removed_point} from Points Type 2")
                    redraw_points()

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)

        plt.show()

        return points_type1, points_type2

    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []
