import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

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
        self.max_possible_score = 0  # Tổng điểm tối đa có thể đạt được
        
        # Thông tin thí sinh
        self.candidate_info = {
            "name": "",
            "email": "",
            "phone": "",
            "position": "",
            "experience_years": 0,
            "education": "",
            "skills": [],
            "summary": "",
            "interview_date": "",
            "interview_duration": 0
        }
        
        # Thời gian bắt đầu phỏng vấn
        self.interview_start_time = None
    
    def extract_candidate_info_from_cv(self):
        """Trích xuất thông tin thí sinh từ CV bằng AI"""
        print("📋 ĐANG TRÍCH XUẤT THÔNG TIN TỪ CV...")
        print("=" * 40)
        
        try:
            # Đọc nội dung CV
            cv_file = "outputs/cv_extracted_text.txt"
            if not os.path.exists(cv_file):
                print("❌ Không tìm thấy file CV. Vui lòng kiểm tra file cv_extracted_text.txt")
                return False
            
            with open(cv_file, 'r', encoding='utf-8') as f:
                cv_content = f.read()
            
            if not cv_content.strip():
                print("❌ File CV trống. Vui lòng kiểm tra nội dung file.")
                return False
            
            # Sử dụng AI để trích xuất thông tin
            extraction_prompt = f"""
            Hãy trích xuất thông tin cá nhân từ CV sau đây:
            
            {cv_content}
            
            Trả về JSON format với các thông tin sau:
            {{
                "name": "Họ và tên đầy đủ",
                "email": "Địa chỉ email nếu có",
                "phone": "Số điện thoại nếu có", 
                "position": "Vị trí ứng tuyển hoặc mục tiêu nghề nghiệp",
                "experience_years": số_năm_kinh_nghiệm,
                "education": "Trường học hoặc bằng cấp",
                "skills": ["kỹ năng 1", "kỹ năng 2", ...],
                "summary": "Tóm tắt ngắn gọn về ứng viên"
            }}
            
            Lưu ý:
            - Nếu không tìm thấy thông tin nào, để trống string hoặc 0 cho số
            - experience_years phải là số nguyên
            - skills phải là array các string
            - Chỉ trả về JSON, không có text khác
            """
            
            response = self.llm.invoke(extraction_prompt)
            
            # Parse JSON response
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    extracted_info = json.loads(json_match.group())
                    
                    # Cập nhật thông tin thí sinh
                    self.candidate_info.update({
                        "name": extracted_info.get("name", ""),
                        "email": extracted_info.get("email", ""),
                        "phone": extracted_info.get("phone", ""),
                        "position": extracted_info.get("position", ""),
                        "experience_years": extracted_info.get("experience_years", 0),
                        "education": extracted_info.get("education", ""),
                        "skills": extracted_info.get("skills", []),
                        "summary": extracted_info.get("summary", "")
                    })
                    
                    # Thêm thông tin thời gian
                    self.candidate_info["interview_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.interview_start_time = datetime.now()
                    
                    print(f"✅ Đã trích xuất thông tin thí sinh:")
                    print(f"   👤 Tên: {self.candidate_info['name']}")
                    print(f"   📧 Email: {self.candidate_info['email']}")
                    print(f"   📱 Phone: {self.candidate_info['phone']}")
                    print(f"   💼 Vị trí: {self.candidate_info['position']}")
                    print(f"   📅 Kinh nghiệm: {self.candidate_info['experience_years']} năm")
                    print(f"   🎓 Học vấn: {self.candidate_info['education']}")
                    print("=" * 50)
                    
                    return True
                else:
                    print("❌ Không thể parse thông tin từ AI response")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"❌ Lỗi parse JSON: {e}")
                print(f"AI Response: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Lỗi khi đọc CV: {e}")
            return False
    
    def collect_candidate_info(self):
        """Thu thập thông tin cơ bản của thí sinh từ CV"""
        success = self.extract_candidate_info_from_cv()
        
        if not success:
            print("⚠️  Không thể trích xuất thông tin từ CV. Chuyển sang nhập thủ công...")
            print("📋 THÔNG TIN THÍ SINH")
            print("=" * 30)
            
            self.candidate_info["name"] = input("👤 Họ và tên: ").strip()
            self.candidate_info["email"] = input("📧 Email: ").strip()
            self.candidate_info["phone"] = input("📱 Số điện thoại: ").strip()
            self.candidate_info["position"] = input("💼 Vị trí ứng tuyển: ").strip()
            
            try:
                years = input("📅 Số năm kinh nghiệm: ").strip()
                self.candidate_info["experience_years"] = int(years) if years else 0
            except ValueError:
                self.candidate_info["experience_years"] = 0
            
            self.candidate_info["interview_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.interview_start_time = datetime.now()
            
            print(f"✅ Đã lưu thông tin thí sinh: {self.candidate_info['name']}")
            print("=" * 50)
        
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
        """Tạo 3 câu hỏi kỹ thuật từ knowledge database dựa trên kiến thức cụ thể"""
        # Tìm kiến thức cụ thể từ knowledge database để tạo câu hỏi
        knowledge_docs = self.knowledge_retriever.get_relevant_documents("kiến thức chuyên môn lý thuyết bài học")
        
        prompt_template = """
        Dựa trên kiến thức chuyên môn sau đây từ tài liệu học tập:
        {knowledge_content}
        
        Hãy tạo 3 câu hỏi kiểm tra kiến thức liên quan đến nhau về:
        - Kiến thức lý thuyết từ tài liệu
        - Khái niệm và định nghĩa
        - Ứng dụng thực tế của kiến thức
        
        Yêu cầu:
        - Câu hỏi phải dựa trực tiếp vào nội dung kiến thức đã cho
        - Kiểm tra khả năng hiểu và áp dụng kiến thức
        - Các câu hỏi phải liên quan đến nhau và cùng chủ đề
        
        Trả về JSON format:
        [
            {{
                "id": 3,
                "question": "Câu hỏi kiến thức 1 (dựa trên nội dung tài liệu)",
                "category": "technical",
                "purpose": "Kiểm tra hiểu biết về khái niệm cơ bản",
                "related_to": "Liên quan đến câu hỏi 4,5 về cùng chủ đề"
            }},
            {{
                "id": 4,
                "question": "Câu hỏi kiến thức 2 (ứng dụng thực tế)",
                "category": "technical", 
                "purpose": "Kiểm tra khả năng áp dụng kiến thức",
                "related_to": "Liên quan đến câu hỏi 3,5"
            }},
            {{
                "id": 5,
                "question": "Câu hỏi kiến thức 3 (phân tích sâu)",
                "category": "technical",
                "purpose": "Kiểm tra khả năng phân tích và đánh giá", 
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
        
        # Thu thập thông tin thí sinh
        self.collect_candidate_info()
        
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
                # Tính điểm trung bình hiện tại
                current_avg = self.total_score / len(self.scores) if self.scores else 0
                if current_avg < 8.0:
                    print(f"\n📊 Điểm trung bình hiện tại: {current_avg:.1f}/10")
                    print("⚠️  Bạn cần đạt điểm trung bình ít nhất 8.0 để tiếp tục câu hỏi sáng tạo.")
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
                self.max_possible_score += 10  # Mỗi câu tối đa 10 điểm
                
                # Tính điểm trung bình hiện tại
                current_avg = self.total_score / len(self.scores)
                
                print(f"📊 Điểm câu này: {score}/10")
                print(f"📈 Tổng điểm: {self.total_score}/{self.max_possible_score}")
                print(f"📊 Điểm trung bình: {current_avg:.1f}/10")
            else:
                print("⚠️  Bạn chưa trả lời. Câu hỏi này sẽ được bỏ qua.")
        
        # Hiển thị kết quả cuối
        self._show_final_results()
        
        # Xuất kết quả ra file JSON
        self.export_interview_results()
    
    def _score_answer(self, question: Dict[str, Any], answer: str) -> float:
        """Chấm điểm câu trả lời bằng Gemini"""
        
        # Lấy context liên quan để chấm điểm
        if question['category'] == 'behavioral' or question['category'] == 'cv_based':
            context_docs = self.cv_retriever.get_relevant_documents(question['question'])
        else:
            context_docs = self.knowledge_retriever.get_relevant_documents(question['question'])
        
        context = "\n".join([doc.page_content for doc in context_docs])
        
        # Xác định tiêu chí chấm điểm dựa trên loại câu hỏi
        if question['category'] == 'technical':
            criteria = """
        1. Kiến thức chính xác (Knowledge): Mức độ hiểu biết đúng về khái niệm
        2. Áp dụng thực tế (Application): Khả năng áp dụng kiến thức vào tình huống thực tế
        3. Phân tích sâu (Analysis): Khả năng phân tích và giải thích chi tiết
        4. Tư duy phản biện (Critical Thinking): Khả năng đánh giá và so sánh
        5. Truyền đạt rõ ràng (Communication): Cách trình bày logic và dễ hiểu
            """
        else:
            criteria = """
        1. Độ chính xác (Correctness): Mức độ lập luận gắn kết với ý chính
        2. Độ bao quát (Coverage): Tỷ lệ phần trăm các ý chính được đề cập  
        3. Lý luận (Reasoning): Cách phân tích từng bước, nêu rõ giả định
        4. Tính sáng tạo (Creativity): Giải pháp mới mẻ nhưng hợp lý
        5. Truyền đạt (Communication): Ngôn ngữ rõ ràng, có cấu trúc
            """
        
        scoring_prompt = f"""
        Bạn là một chuyên gia phỏng vấn nhân sự. Hãy chấm điểm câu trả lời một cách công bằng và chính xác.

        Câu hỏi: {question['question']}
        Loại câu hỏi: {question['category']}
        Mục đích: {question['purpose']}
        Câu trả lời: {answer}
        Context liên quan: {context}

        Chấm điểm theo thang điểm 10 cho từng tiêu chí:
        {criteria}

        Yêu cầu:
        - Tổng điểm phải là trung bình của 5 tiêu chí (không cộng dồn)
        - Điểm từ 0-10 cho mỗi tiêu chí
        - Đánh giá dựa trên chất lượng thực tế của câu trả lời

        Trả về JSON format:
        {{
            "criteria_1": điểm_số,
            "criteria_2": điểm_số,
            "criteria_3": điểm_số,
            "criteria_4": điểm_số,
            "criteria_5": điểm_số,
            "total": tổng_điểm_trung_bình,
            "feedback": "Nhận xét chi tiết về câu trả lời"
        }}
        """
        
        response = self.llm.invoke(scoring_prompt)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                score_data = json.loads(json_match.group())
                total_score = float(score_data.get('total', 0))
                
                # Đảm bảo điểm số trong khoảng hợp lệ (0-10)
                if 0 <= total_score <= 10:
                    return total_score
                else:
                    # Nếu điểm không hợp lệ, tính trung bình từ các tiêu chí
                    criteria_scores = [
                        score_data.get('criteria_1', 0),
                        score_data.get('criteria_2', 0),
                        score_data.get('criteria_3', 0),
                        score_data.get('criteria_4', 0),
                        score_data.get('criteria_5', 0)
                    ]
                    valid_scores = [float(s) for s in criteria_scores if isinstance(s, (int, float)) and 0 <= s <= 10]
                    if valid_scores:
                        return sum(valid_scores) / len(valid_scores)
                    
        except Exception as e:
            print(f"⚠️ Lỗi khi chấm điểm: {e}")
        
        # Trả về điểm mặc định nếu có lỗi
        return 5.0
    
    def export_interview_results(self):
        """Xuất kết quả phỏng vấn ra file JSON"""
        try:
            # Tính thời gian phỏng vấn
            if self.interview_start_time:
                interview_duration = (datetime.now() - self.interview_start_time).total_seconds() / 60
                self.candidate_info["interview_duration"] = round(interview_duration, 2)
            
            # Tạo cấu trúc dữ liệu hoàn chỉnh
            avg_score = self.total_score / len(self.scores) if self.scores else 0
            total_possible_all_questions = len(self.questions) * 10  # Tất cả câu hỏi × 10 điểm
            
            interview_data = {
                "candidate_info": self.candidate_info,
                "interview_summary": {
                    "total_questions": len(self.questions),
                    "total_answers": len(self.answers),
                    "total_score": self.total_score,
                    "max_possible_score": self.max_possible_score,
                    "total_possible_score_all_questions": total_possible_all_questions,
                    "average_score": round(avg_score, 2),
                    "interview_status": self._get_interview_status(),
                    "candidate_education": self.candidate_info.get("education", ""),
                    "candidate_skills": self.candidate_info.get("skills", []),
                    "candidate_summary": self.candidate_info.get("summary", "")
                },
                "questions_and_answers": [],
                "detailed_scores": [],
                "export_info": {
                    "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "system_version": "1.0.0"
                }
            }
            
            # Thêm chi tiết câu hỏi và câu trả lời
            for i, (question, answer, score) in enumerate(zip(self.questions, self.answers, self.scores)):
                qa_detail = {
                    "question_id": question['id'],
                    "question_category": question['category'],
                    "question": question['question'],
                    "question_purpose": question['purpose'],
                    "question_related_to": question.get('related_to', ''),
                    "answer": answer,
                    "score": score,
                    "max_score": 10
                }
                interview_data["questions_and_answers"].append(qa_detail)
                
                # Thêm chi tiết điểm số (nếu có)
                score_detail = {
                    "question_id": question['id'],
                    "score": score,
                    "percentage": round((score / 10) * 100, 1)
                }
                interview_data["detailed_scores"].append(score_detail)
            
            # Tạo thư mục lưu trữ
            output_dir = Path("interview_results")
            output_dir.mkdir(exist_ok=True)
            
            # Tạo tên file dựa trên tên thí sinh và ngày
            safe_name = "".join(c for c in self.candidate_info["name"] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_name}_{timestamp}.json"
            filepath = output_dir / filename
            
            # Xuất file JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(interview_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 Đã xuất kết quả phỏng vấn ra file: {filepath}")
            print(f"📁 Thư mục lưu trữ: {output_dir.absolute()}")
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Lỗi khi xuất file: {e}")
            return None
    
    def _get_interview_status(self):
        """Xác định trạng thái phỏng vấn dựa trên điểm trung bình"""
        if not self.scores:
            return "Chưa có điểm"
        
        avg_score = self.total_score / len(self.scores)
        if avg_score >= 8.0:
            return "Xuất sắc - Đạt yêu cầu cao"
        elif avg_score >= 6.0:
            return "Tốt - Đáp ứng yêu cầu"
        elif avg_score >= 4.0:
            return "Trung bình - Cần cải thiện"
        else:
            return "Chưa đạt - Cần trau dồi thêm"
    
    def _show_final_results(self):
        """Hiển thị kết quả cuối cùng"""
        print("\n" + "=" * 50)
        print("📊 KẾT QUẢ PHỎNG VẤN")
        print("=" * 50)
        
        if not self.scores:
            print("❌ Không có câu trả lời nào được chấm điểm.")
            return
        
        avg_score = self.total_score / len(self.scores)
        total_possible_all = len(self.questions) * 10
        
        print(f"🎯 Tổng điểm đạt được: {self.total_score}/{self.max_possible_score}")
        print(f"📊 Điểm trung bình: {avg_score:.1f}/10")
        print(f"📈 Số câu đã trả lời: {len(self.answers)}/{len(self.questions)}")
        print(f"🏆 Tổng điểm tối đa (nếu trả lời hết): {total_possible_all}")
        
        # Hiển thị tỷ lệ phần trăm
        if self.max_possible_score > 0:
            percentage = (self.total_score / self.max_possible_score) * 100
            print(f"📊 Tỷ lệ thành tích: {percentage:.1f}%")
        
        if avg_score >= 8.0:
            print("🎉 XUẤT SẮC! Bạn đã vượt qua phỏng vấn với điểm số cao.")
        elif avg_score >= 6.0:
            print("👍 TỐT! Bạn có khả năng đáp ứng yêu cầu công việc.")
        elif avg_score >= 4.0:
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
