import clang
from clang.cindex import Config
from src.config.config import CLANG_PATH

Config.set_library_path(CLANG_PATH.UBUNTU)

def check_cpp_code_ast(code: str) -> bool:
    """
    Kiểm tra xem đoạn code C++ có parse được sang AST không.

    Trả về True nếu code parse thành công (không có lỗi nghiêm trọng),
    ngược lại trả về False.
    """
    # Tạo một instance của Index
    index = clang.cindex.Index.create()
    try:
        # Parse code dưới dạng một file tạm thời "tmp.cpp"
        tu = index.parse(
            'tmp.cpp',
            args=['-std=c++11'],  # Bạn có thể thay đổi options nếu cần
            unsaved_files=[('tmp.cpp', code)],
            options=0
        )

        # Kiểm tra các thông báo chẩn đoán: nếu có lỗi (Diagnostic.Error trở lên), báo lỗi
        errors = [diag for diag in tu.diagnostics if diag.severity >= clang.cindex.Diagnostic.Error]
        if errors:
            print("Có lỗi khi parse code:")
            for error in errors:
                print(" -", error)
            return False

        # Nếu không có lỗi, trả về True
        return True

    except Exception as e:
        print("Có ngoại lệ xảy ra:", e)
        return False