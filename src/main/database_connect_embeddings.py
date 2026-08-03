from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


def get_psql_session():
    engine = create_engine("postgresql://postgres:postgres@localhost/text_embeddings")
    Base.metadata.create_all(engine)

    # Create a session
    Session = sessionmaker(bind=engine)
    return Session()


# //print("yo")
# //print(get_psql_session())


class TextEmbedding(Base):
    __tablename__ = "text_embeddings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding = Column(Vector)
    content = Column(String)
    file_name = Column(String)
    sentence_number = Column(Integer)

    def __str__(self):
        return self.content + " " + str(self.id)

    @classmethod
    def truncate(cls, session):
        session.execute(text(f"TRUNCATE TABLE {cls.__tablename__} RESTART IDENTITY CASCADE"))
        session.commit()

    @classmethod
    def delete_by_file_name(cls, session, file_name):
        session.query(cls).filter(cls.file_name == file_name).delete(synchronize_session=False)
