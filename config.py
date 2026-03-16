import os
from dotenv import load_dotenv

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_CONTAINER_RAW = os.getenv("AZURE_CONTAINER_RAW", "ngsim-raw")
AZURE_CONTAINER_OUTPUT = os.getenv("AZURE_CONTAINER_OUTPUT", "ngsim-output")

NGSIM_DATA_PATH = os.getenv("NGSIM_DATA_PATH", "data/trajectories-0750am-0805am.csv")
WINDOW_SIZE_SEC = int(os.getenv("WINDOW_SIZE_SEC", "5"))
SAMPLING_RATE_HZ = int(os.getenv("SAMPLING_RATE_HZ", "10"))
FRAMES_PER_WINDOW = WINDOW_SIZE_SEC * SAMPLING_RATE_HZ

NGSIM_COLUMNS = [
    "Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time",
    "Local_X", "Local_Y", "Global_X", "Global_Y",
    "v_Length", "v_Width", "v_Class",
    "v_Vel", "v_Acc", "Lane_ID",
    "Preceding", "Following",
    "Space_Hdwy", "Time_Hdwy"
]

CF_MAX_SPACE_HEADWAY_FT = 200.0
CF_MIN_VELOCITY_FT_S = 5.0
CF_MAX_SPEED_DIFF_FT_S = 15.0

SG_DECEL_THRESHOLD_FT_S2 = -5.0
SG_ACCEL_THRESHOLD_FT_S2 = 3.0
SG_LOW_SPEED_THRESHOLD_FT_S = 10.0

LC_LANE_MAINLINE = [1, 2, 3, 4, 5]
LC_SURROUNDING_RADIUS_FT = 200.0

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
