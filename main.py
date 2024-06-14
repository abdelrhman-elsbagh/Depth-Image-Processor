import os
import dask.dataframe as dd
import pandas as pd
import cv2
import numpy as np
import pymongo
import gridfs
from io import BytesIO
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define file path using current directory
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'img.csv')

# Create a directory to serve images
images_dir = os.path.join(current_dir, 'images')
os.makedirs(images_dir, exist_ok=True)


class ImageProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.depths = None
        self.resized_image = None

    def analyze_csv(self):
        data = pd.read_csv(self.file_path)
        data_info = data.info()
        data_head = data.head()
        depth_range = (data['depth'].min(), data['depth'].max())
        logger.info(f"CSV info: {data_info}")
        logger.info(f"CSV head: {data_head}")
        logger.info(f"Depth range: {depth_range[0]} to {depth_range[1]}")
        return depth_range

    def load_and_process_data(self):
        try:
            # Specify the dtype for all columns to avoid dtype inference issues
            dtype = {f'col{i}': 'float64' for i in range(1, 201)}
            dtype['depth'] = 'float64'

            # Load the CSV file using Dask
            dask_df = dd.read_csv(self.file_path, dtype=dtype)
            logger.info("CSV file loaded successfully with Dask.")

            # Handle NaN values by filling with a specified value (e.g., 0)
            dask_df = dask_df.fillna(0)
            logger.info("NaN values handled.")

            # Convert Dask dataframe to Pandas dataframe for processing
            data_filled = dask_df.compute()
            logger.info("Dask dataframe computed successfully.")

            # Drop the 'depth' column and convert to numpy array
            self.depths = data_filled['depth'].values
            image_data = data_filled.drop(columns=['depth']).values

            # Resize the image data to 150 columns
            new_width = 150
            self.resized_image = cv2.resize(image_data, (new_width, image_data.shape[0]), interpolation=cv2.INTER_AREA)
            logger.info("Image resized successfully.")

            # Save the resized image for verification
            cv2.imwrite(os.path.join(images_dir, 'resized_image.jpg'), self.resized_image)
            logger.info("Resized image saved successfully.")

            # Store the resized image in MongoDB
            self.store_image_in_db(image_data.shape, self.resized_image.shape)

        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def store_image_in_db(self, original_shape, new_shape):
        try:
            # Connect to MongoDB
            client = pymongo.MongoClient('mongodb://localhost:27017/')
            db = client['image_db']
            fs = gridfs.GridFS(db)
            logger.info("Connected to MongoDB.")

            # Convert the resized image to binary format (JPEG)
            is_success, buffer = cv2.imencode(".jpg", self.resized_image)
            io_buf = BytesIO(buffer)

            # Store the image in MongoDB using GridFS
            file_id = fs.put(
                io_buf.getvalue(),
                filename='resized_image.jpg',
                original_shape=original_shape,
                new_shape=new_shape
            )
            logger.info(f"Stored image with file_id: {file_id}")

        except Exception as e:
            logger.error(f"An error occurred while storing the image in MongoDB: {e}")

    def get_image_frame(self, depth_min, depth_max):
        logger.info(f"Received depth_min: {depth_min}, depth_max: {depth_max}")
        mask = (self.depths >= depth_min) & (self.depths <= depth_max)
        if not np.any(mask):
            logger.warning("No data found in the specified depth range.")
            raise ValueError("No data found in the specified depth range.")

        frame = self.resized_image[mask, :]
        frame_image_path = os.path.join(images_dir, 'frame_image.jpg')
        cv2.imwrite(frame_image_path, frame)
        return frame_image_path

    def get_colored_image_frame(self, depth_min, depth_max, color_map):
        frame_path = self.get_image_frame(depth_min, depth_max)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        # Apply a custom color map
        color_map_code = self.get_color_map_code(color_map)
        colored_frame = cv2.applyColorMap(frame, color_map_code)
        colored_image_path = os.path.join(images_dir, 'colored_frame.jpg')
        cv2.imwrite(colored_image_path, colored_frame)
        return colored_image_path

    @staticmethod
    def get_color_map_code(color_map):
        color_maps = {
            'autumn': cv2.COLORMAP_AUTUMN,
            'bone': cv2.COLORMAP_BONE,
            'jet': cv2.COLORMAP_JET,
            'winter': cv2.COLORMAP_WINTER,
            'rainbow': cv2.COLORMAP_RAINBOW,
            'ocean': cv2.COLORMAP_OCEAN,
            'summer': cv2.COLORMAP_SUMMER,
            'spring': cv2.COLORMAP_SPRING,
            'cool': cv2.COLORMAP_COOL,
            'hsv': cv2.COLORMAP_HSV,
            'pink': cv2.COLORMAP_PINK,
            'hot': cv2.COLORMAP_HOT,
            'parula': cv2.COLORMAP_PARULA
        }
        return color_maps.get(color_map, cv2.COLORMAP_JET)


image_processor = ImageProcessor(file_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    image_processor.analyze_csv()
    image_processor.load_and_process_data()
    yield
    # Clean up resources here if needed


app = FastAPI(lifespan=lifespan)


class DepthRange(BaseModel):
    depth_min: float
    depth_max: float


class ColoredDepthRange(DepthRange):
    color_map: str


@app.post("/get-image-frame/")
async def get_image_frame(request: Request, depth_range: DepthRange):
    try:
        frame_image_path = image_processor.get_image_frame(depth_range.depth_min, depth_range.depth_max)
        base_url = request.base_url
        frame_image_url = f"{base_url}images/frame_image.jpg"
        return {"frame_image_url": frame_image_url}
    except ValueError as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-colored-image-frame/")
async def get_colored_image_frame(request: Request, depth_range: ColoredDepthRange):
    try:
        colored_image_path = image_processor.get_colored_image_frame(
            depth_range.depth_min, depth_range.depth_max, depth_range.color_map
        )
        base_url = request.base_url
        colored_image_url = f"{base_url}images/colored_frame.jpg"
        return {"colored_image_url": colored_image_url}
    except ValueError as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = os.path.join(images_dir, image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
