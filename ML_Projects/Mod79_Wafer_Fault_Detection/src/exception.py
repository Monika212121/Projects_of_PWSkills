import sys

def get_err_msg_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_msg = "Error occurred in Python script name [{0}], line no. [{1}], error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_msg



class CustomException(Exception):

    def __init__(self, error_msg, error_detail: sys):
        """ :param error mesage: error message in string format """

        super().__init__(error_msg)

        self.error_msg = get_err_msg_detail(
            error_msg, 
            error_detail = error_detail
        )

    def __str__(self):
        return self.error_msg