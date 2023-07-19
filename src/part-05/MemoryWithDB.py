from langchain.memory import MongoDBChatMessageHistory

# Provide the connection string to connect to the MongoDB database
connection_string = "mongodb://localhost:27017/chat_history"


message_history = MongoDBChatMessageHistory(
    connection_string=connection_string, session_id="dGVzdC1zZXNzaW9u"
)

message_history.add_user_message("hi!")
message_history.add_ai_message("whats up?")