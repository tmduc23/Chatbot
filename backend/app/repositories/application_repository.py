"""
MỤC ĐÍCH: Mọi truy vấn database liên quan đến bảng "application"

Repository = lớp duy nhất được viết câu query. Service gọi repository, 
không tự viết select(...) rải rác -> dễ tìm, dễ sửa, dễ kiểm soát lớp bảo mật.

Repository KHÔNG commit - việc commit thuộc về service (người biết khi nào một nghiệp vụ hoàn tất)

                Application Repository
                         │
              ┌──────────┴──────────┐
              │                     │
            get()                create()
              │                     │
    Tìm Application đã có       Tạo Application mới
              │                     │
              ▼                     ▼
           SELECT                 INSERT
              │                     │
              └──────────┬──────────┘
                         ▼
                    PostgreSQL
"""


from sqlalchemy.orm import Session
from app.models import Application



"""
    MỤC ĐÍCH:
    Lấy một Application từ Database dựa vào application_id.
    Ví dụ:
        application_id = "t_cv"
    Hàm sẽ tìm trong bảng `applications`:
        id = "t_cv"
    Kết quả:
        - Tìm thấy  → trả về object Application
        - Không tìm thấy → trả về None
"""

# Lấy application theo khóa chính bằng db.get
def get(db: Session, application_id: str) -> Application | None:
    # db.get() tìm một object dựa trên Primary Key
    return db.get(Application, application_id)


"""
    MỤC ĐÍCH:
    Tạo một Application mới và đưa nó vào Database trong transaction hiện tại

    Ví dụ:
        create(
            db,
            "t_cv",
            "CV Chatbot",
            "Chatbot hỏi đáp về CV"
        )
    Hàm sẽ tạo:
        Application(
            id="t_cv",
            name="CV Chatbot",
            description="Chatbot hỏi đáp về CV"
        )
"""

# Thêm application rồi flush, không commit
def create(db: Session, application_id: str, name: str, description: str | None = None) -> Application:
    app = Application(id=application_id, name=name, description=description)
    db.add(app)
    db.flush()
    return app