import logging

from src.models.codet5_model import Codet5Model
from src.utils.ast_parser import check_cpp_code_ast

if __name__ == '__main__':
    model = Codet5Model("codet5-base")
    res = model.generate_from_sample("int sum(int a, int b) {return a + b;}")
    logging.info(check_cpp_code_ast(res))