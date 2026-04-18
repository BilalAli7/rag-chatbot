import os
from typing import Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class ChatEngine:
    def __init__(self, vectorstore):
        if vectorstore is None:
            raise ValueError("Vectorstore cannot be None!")

        self.vectorstore = vectorstore
        self.chat_history = []
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        print("🤖 Initializing Groq model...")
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.7,
        )
        print("✅ Model ready!")

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _format_history(self):
        lines = []
        for msg in self.chat_history:
            if isinstance(msg, HumanMessage):
                lines.append(f"Human: {msg.content}")
            else:
                lines.append(f"Assistant: {msg.content}")
        return "\n".join(lines)

    def ask(self, question: str) -> Dict:
        try:
            print(f"\n💭 Thinking about: {question}")

            # Retrieve relevant docs
            docs = self.retriever.invoke(question)
            context = self._format_docs(docs)
            history = self._format_history()

            # Build prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant. Use the context below to answer the question.
If you don't know the answer, say so — don't make one up.

Context:
{context}

Chat History:
{history}"""),
                ("human", "{question}"),
            ])

            chain = prompt | self.llm | StrOutputParser()

            answer = chain.invoke({
                "context": context,
                "history": history,
                "question": question,
            })

            # Update history
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))

            print("✅ Answer generated!")
            return {"answer": answer, "sources": docs, "success": True}

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {"answer": f"Sorry, I encountered an error: {str(e)}", "sources": [], "success": False}

    def clear_history(self):
        self.chat_history = []
        print("🗑️ Conversation history cleared!")