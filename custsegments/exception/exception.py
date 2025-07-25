import sys
import traceback
# Corrected import path assuming customer_segments is the root for execution context or PYTHONPATH is set
# If run from within customer_segments, then custsegments.logging.logger is correct
# For now, assuming custsegments is a package within customer_segments
from custsegments.logging.logger import logging 

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None: # Handle cases where exc_info might not return full tuple
        return f"Error occurred: {str(error)}"
        
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message_str: str, error_detail: sys):
        """
        :param error_message_str: error message in string format
        """
        super().__init__(error_message_str)
        self.error_message = error_message_detail(error_message_str, error_detail=error_detail)

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by zero error")
        raise CustomException(str(e), sys)
