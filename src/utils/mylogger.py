import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # Handler để in log ra console
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)