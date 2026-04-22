from sqlalchemy import Column, Integer, String, Text
from database import Base

class UserProfile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    goal = Column(String)
    constraints = Column(String)
    generated_plan = Column(Text)