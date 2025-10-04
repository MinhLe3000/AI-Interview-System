import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from GetApikey import loadapi


class InterviewSystem:
    def __init__(self):
        """Khởi tạo hệ thống phỏng vấn với 2 vector database"""
        self.api_key = loadapi()
        
        # Khởi tạo embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Load vector databases
        self.cv_db = FAISS.load_local(
            "vector_db_cv", 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.knowledge_db = FAISS.load_local(
            "vector_db2chunk_nltk", 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Khởi tạo Gemini LLM
        self.llm = GoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.7
        )
        
        # Khởi tạo retriever
        self.cv_retriever = self.cv_db.as_retriever(search_kwargs={"k": 3})
        self.knowledge_retriever = self.knowledge_db.as_retriever(search_kwargs={"k": 3})
        
        # Lưu trữ câu hỏi và điểm số
        self.questions = []
        self.answers = []
        self.scores = []
        self.total_score = 0
        
    def generate_questions(self) -> List[Dict[str, Any]]:
        """Tạo 8 câu hỏi phỏng vấn từ 2 vector database"""
        
        # 1. Tạo 2 câu hỏi hành vi từ CV
        behavioral_questions = self._generate_behavioral_questions()
        
        # 2. Tạo 3 câu hỏi kỹ thuật từ knowledge database
        technical_questions = self._generate_technical_questions()
        
        # 3. Tạo 2 câu hỏi về dự án/kinh nghiệm từ CV
        project_questions = self._generate_project_questions()
        
        # 4. Tạo 1 câu hỏi sáng tạo (sẽ hiển thị sau khi điểm > 8)
        creative_question = self._generate_creative_question()
        
        # Gộp tất cả câu hỏi
        all_questions = behavioral_questions + technical_questions + project_questions
        all_questions.append(creative_question)
        
        self.questions = all_questions
        return all_questions
    
    def _generate_behavioral_questions(self) -> List[Dict[str, Any]]:
        """Tạo 2 câu hỏi hành vi từ CV database"""
        # Tìm thông tin về kỹ năng mềm, kinh nghiệm làm việc nhóm
        cv_docs = self.cv_retriever.get_relevant_documents("kỹ năng giao tiếp làm việc nhóm thách thức động lực")
        
        prompt_template = """
        Dựa trên thông tin CV sau:
        {cv_content}
        
        Hãy tạo 2 câu hỏi hành vi (behavioral) liên quan đến nhau về:
        - Làm việc nhóm
        - Xử lý thách thức
        - Động lực làm việc
        
        Trả về JSON format:
        [
            {{
                "id": 1,
                "question": "Câu hỏi 1",
                "category": "behavioral",
                "purpose": "Mục đích đánh giá",
                "related_to": "Câu hỏi liên quan đến câu hỏi 2"
            }},
            {{
                "id": 2,
                "question": "Câu hỏi 2", 
                "category": "behavioral",
                "purpose": "Mục đích đánh giá",
                "related_to": "Câu hỏi liên quan đến câu hỏi 1"
            }}
        ]
        """
        
        cv_content = "\n".join([doc.page_content for doc in cv_docs])
        prompt = PromptTemplate(template=prompt_template, input_variables=["cv_content"])
        formatted_prompt = prompt.format(cv_content=cv_content)
        
        response = self.llm.invoke(formatted_prompt)
        return self._parse_json_response(response)
    
    def _generate_technical_questions(self) -> List[Dict[str, Any]]:
        """Tạo 3 câu hỏi kỹ thuật từ knowledge database"""
        # Tìm thông tin kỹ thuật liên quan ( cần sửa lại)
        knowledge_docs = self.knowledge_retriever.get_relevant_documents("kiến thức kỹ thuật lập trình công nghệ")
        
        prompt_template = """
        Dựa trên kiến thức kỹ thuật sau:
        {knowledge_content}
        
        Hãy tạo 3 câu hỏi kỹ thuật liên quan đến nhau về:
        - Kiến thức lập trình
        - Công nghệ và framework
        - Best practices
        
        Trả về JSON format:
        [
            {{
                "id": 3,
                "question": "Câu hỏi kỹ thuật 1",
                "category": "technical",
                "purpose": "Mục đích đánh giá",
                "related_to": "Liên quan đến câu hỏi 4,5"
            }},
            {{
                "id": 4,
                "question": "Câu hỏi kỹ thuật 2",
                "category": "technical", 
                "purpose": "Mục đích đánh giá",
                "related_to": "Liên quan đến câu hỏi 3,5"
            }},
            {{
                "id": 5,
                "question": "Câu hỏi kỹ thuật 3",
                "category": "technical",
                "purpose": "Mục đích đánh giá", 
                "related_to": "Liên quan đến câu hỏi 3,4"
            }}
        ]
        """
        
        knowledge_content = "\n".join([doc.page_content for doc in knowledge_docs])
        prompt = PromptTemplate(template=prompt_template, input_variables=["knowledge_content"])
        formatted_prompt = prompt.format(knowledge_content=knowledge_content)
        
        response = self.llm.invoke(formatted_prompt)
        return self._parse_json_response(response)
    
    def _generate_project_questions(self) -> List[Dict[str, Any]]:
        """Tạo 2 câu hỏi về dự án/kinh nghiệm từ CV"""
        # Tìm thông tin về dự án và kinh nghiệm
        cv_docs = self.cv_retriever.get_relevant_documents("dự án kinh nghiệm thành tích hoạt động")
        
        prompt_template = """
        Dựa trên thông tin CV về dự án và kinh nghiệm:
        {cv_content}
        
        Hãy tạo 2 câu hỏi cụ thể về dự án/kinh nghiệm liên quan đến nhau:
        - Dự án đã tham gia
        - Kinh nghiệm làm việc
        - Thành tích đạt được
        
        Trả về JSON format:
        [
            {{
                "id": 6,
                "question": "Câu hỏi về dự án 1",
                "category": "cv_based",
                "purpose": "Mục đích đánh giá",
                "related_to": "Liên quan đến câu hỏi 7"
            }},
            {{
                "id": 7,
                "question": "Câu hỏi về dự án 2",
                "category": "cv_based",
                "purpose": "Mục đích đánh giá",
                "related_to": "Liên quan đến câu hỏi 6"
            }}
        ]
        """
        
        cv_content = "\n".join([doc.page_content for doc in cv_docs])
        prompt = PromptTemplate(template=prompt_template, input_variables=["cv_content"])
        formatted_prompt = prompt.format(cv_content=cv_content)
        
        response = self.llm.invoke(formatted_prompt)
        return self._parse_json_response(response)
    
    def _generate_creative_question(self) -> Dict[str, Any]:
        """Tạo 1 câu hỏi sáng tạo kết hợp cả 2 database"""
        # Lấy thông tin từ cả 2 database
        cv_docs = self.cv_retriever.get_relevant_documents("kỹ năng kinh nghiệm")
        knowledge_docs = self.knowledge_retriever.get_relevant_documents("giải quyết vấn đề tư duy phản biện")
        
        prompt_template = """
        Dựa trên thông tin CV và kiến thức kỹ thuật:
        CV: {cv_content}
        Knowledge: {knowledge_content}
        
        Hãy tạo 1 câu hỏi sáng tạo/tình huống giả định để kiểm tra:
        - Khả năng giải quyết vấn đề
        - Tư duy phản biện
        - Sáng tạo trong công việc
        
        Trả về JSON format:
        {{
            "id": 8,
            "question": "Câu hỏi sáng tạo",
            "category": "creative",
            "purpose": "Kiểm tra khả năng giải quyết vấn đề và tư duy phản biện",
            "related_to": "Kết hợp kiến thức từ CV và technical knowledge"
        }}
        """
        
        cv_content = "\n".join([doc.page_content for doc in cv_docs])
        knowledge_content = "\n".join([doc.page_content for doc in knowledge_docs])
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["cv_content", "knowledge_content"])
        formatted_prompt = prompt.format(cv_content=cv_content, knowledge_content=knowledge_content)
        
        response = self.llm.invoke(formatted_prompt)
        result = self._parse_json_response(response)
        return result[0] if result else {}
    
    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON response từ LLM"""
        try:
            # Tìm JSON trong response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response)
        except:
            return []
    
    def conduct_interview(self):
        """Tiến hành phỏng vấn"""
        print("🎯 HỆ THỐNG PHỎNG VẤN THÔNG MINH")
        print("=" * 50)
        
        # Tạo câu hỏi
        print("📝 Đang tạo câu hỏi phỏng vấn...")
        questions = self.generate_questions()
        
        if not questions:
            print("❌ Không thể tạo câu hỏi. Vui lòng kiểm tra vector database.")
            return
        
        print(f"✅ Đã tạo {len(questions)} câu hỏi")
        print("\n" + "=" * 50)
        
        # Hiển thị và thu thập câu trả lời
        for i, question in enumerate(questions):
            if i == 7:  # Câu hỏi sáng tạo
                if self.total_score < 8:
                    print(f"\n📊 Điểm hiện tại: {self.total_score}/10")
                    print("⚠️  Bạn cần đạt ít nhất 8 điểm để tiếp tục câu hỏi sáng tạo.") # Hãy sửa lại phần này, không cần phải phải in ra
                    break
            
            print(f"\n❓ Câu hỏi {question['id']} ({question['category']}):")
            print(f"   {question['question']}")
            print(f"   Mục đích: {question['purpose']}")
            
            answer = input("\n💬 Câu trả lời của bạn: ")
            
            if answer.strip():
                # Chấm điểm câu trả lời
                score = self._score_answer(question, answer)
                self.answers.append(answer)
                self.scores.append(score)
                self.total_score += score
                
                print(f"📊 Điểm câu này: {score}/10")
                print(f"📈 Tổng điểm: {self.total_score}/10")
            else:
                print("⚠️  Bạn chưa trả lời. Câu hỏi này sẽ được bỏ qua.")
        
        # Hiển thị kết quả cuối
        self._show_final_results()
    
    def _score_answer(self, question: Dict[str, Any], answer: str) -> float:
        """Chấm điểm câu trả lời bằng Gemini"""
        
        # Lấy context liên quan để chấm điểm
        if question['category'] == 'behavioral' or question['category'] == 'cv_based':
            context_docs = self.cv_retriever.get_relevant_documents(question['question'])
        else:
            context_docs = self.knowledge_retriever.get_relevant_documents(question['question'])
        
        context = "\n".join([doc.page_content for doc in context_docs])
        
        scoring_prompt = f"""
        Bạn là một chuyên gia phỏng vấn nhân sự. Hãy chấm điểm câu trả lời theo 5 tiêu chí:

        Câu hỏi: {question['question']}
        Câu trả lời: {answer}
        Context liên quan: {context}

        Chấm điểm theo thang điểm 10 cho từng tiêu chí:

        1. Độ chính xác (Correctness): Mức độ lập luận gắn kết với ý chính
        2. Độ bao quát (Coverage): Tỷ lệ phần trăm các ý chính được đề cập  
        3. Lý luận (Reasoning): Cách phân tích từng bước, nêu rõ giả định
        4. Tính sáng tạo (Creativity): Giải pháp mới mẻ nhưng hợp lý
        5. Truyền đạt (Communication): Ngôn ngữ rõ ràng, có cấu trúc

        Trả về JSON format:
        {{
            "correctness": điểm_số,
            "coverage": điểm_số,
            "reasoning": điểm_số,
            "creativity": điểm_số,
            "communication": điểm_số,
            "total": tổng_điểm,
            "feedback": "Nhận xét chi tiết về câu trả lời"
        }}
        """
        
        response = self.llm.invoke(scoring_prompt)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                score_data = json.loads(json_match.group())
                return float(score_data.get('total', 0))
        except:
            pass
        
        return 0.0
    
    def _show_final_results(self):
        """Hiển thị kết quả cuối cùng"""
        print("\n" + "=" * 50)
        print("📊 KẾT QUẢ PHỎNG VẤN")
        print("=" * 50)
        
        print(f"🎯 Tổng điểm: {self.total_score}/10")
        
        if self.total_score >= 8:
            print("🎉 XUẤT SẮC! Bạn đã vượt qua phỏng vấn với điểm số cao.")
        elif self.total_score >= 6:
            print("👍 TỐT! Bạn có khả năng đáp ứng yêu cầu công việc.")
        elif self.total_score >= 4:
            print("⚠️  TRUNG BÌNH. Bạn cần cải thiện thêm một số kỹ năng.")
        else:
            print("❌ CHƯA ĐẠT. Bạn cần trau dồi thêm kiến thức và kỹ năng.")
        
        # Hiển thị chi tiết từng câu
        print("\n📋 CHI TIẾT TỪNG CÂU:")
        for i, (question, answer, score) in enumerate(zip(self.questions, self.answers, self.scores)):
            print(f"\nCâu {question['id']} ({question['category']}): {score}/10")
            print(f"  Hỏi: {question['question']}")
            print(f"  Trả lời: {answer[:100]}...")


def main():
    """Hàm main để chạy hệ thống phỏng vấn"""
    try:
        interview_system = InterviewSystem()
        interview_system.conduct_interview()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("Vui lòng kiểm tra:")
        print("1. Vector database đã được tạo chưa")
        print("2. API key Gemini có hợp lệ không")
        print("3. Các thư viện đã được cài đặt đầy đủ")


if __name__ == "__main__":
    main()
