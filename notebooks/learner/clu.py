import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class OperationalResult(BaseModel):
    answer: str = Field(
        ..., description="The answer to the input query based on current knowledge."
    )
    explanation: str = Field(..., description="The reasoning behind the answer.")


class RuleModification(BaseModel):
    knowledge_id: int = Field(..., description="The knowledge item ID to modify.")
    new_content: str = Field(..., description="The new content for the knowledge item.")


class FeedbackResult(BaseModel):
    feedback: str = Field(..., description="Detailed feedback on the previous answer.")
    retain_knowledge_ids: List[int] = Field(
        ..., description="IDs of knowledge items to retain as they are."
    )
    modify_knowledge: List[RuleModification] = Field(
        default_factory=list, description="List of knowledge items to modify."
    )
    add_positive_knowledge: List[str] = Field(
        default_factory=list, description="New positive knowledge items to add."
    )
    add_negative_knowledge: List[str] = Field(
        default_factory=list, description="New negative knowledge items to add."
    )
    remove_knowledge_ids: List[int] = Field(
        default_factory=list, description="IDs of knowledge items to remove."
    )


class KnowledgeItem(BaseModel):
    id: int = Field(..., description="Unique identifier for the knowledge item.")
    content: str = Field(..., description="The content of the knowledge item.")
    type: str = Field(..., description="Type of knowledge: 'positive' or 'negative'.")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp."
    )
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp."
    )


class Memory:
    def __init__(self, file_path: str = "memory.json"):
        self.file_path = Path(file_path)
        self.positive_knowledge: List[KnowledgeItem] = []
        self.negative_knowledge: List[KnowledgeItem] = []
        self.next_knowledge_id = 1
        self._load()

    def _load(self):
        if not self.file_path.exists():
            self._save()
            return

        data = json.loads(self.file_path.read_text())
        self.next_knowledge_id = data.get("next_knowledge_id", 1)
        self.positive_knowledge = [
            KnowledgeItem(**item) for item in data.get("positive_knowledge", [])
        ]
        self.negative_knowledge = [
            KnowledgeItem(**item) for item in data.get("negative_knowledge", [])
        ]

    def _save(self):
        data = {
            "next_knowledge_id": self.next_knowledge_id,
            "positive_knowledge": [
                item.model_dump() for item in self.positive_knowledge
            ],
            "negative_knowledge": [
                item.model_dump() for item in self.negative_knowledge
            ],
        }
        # Ensure datetime objects are serialized as strings
        self.file_path.write_text(json.dumps(data, default=str, indent=2))

    def get_positive_knowledge_context(self) -> str:
        if not self.positive_knowledge:
            return "No positive knowledge available."
        return "Positive Knowledge (use IDs to refer):\n" + "\n".join(
            f"{item.id}. {item.content}" for item in self.positive_knowledge
        )

    def get_negative_knowledge_context(self) -> str:
        if not self.negative_knowledge:
            return "No negative knowledge available."
        return "Negative Knowledge (use IDs to refer):\n" + "\n".join(
            f"{item.id}. {item.content}" for item in self.negative_knowledge
        )

    def update_from_feedback(self, feedback: FeedbackResult):
        # Retain specified knowledge items
        retain_ids = set(feedback.retain_knowledge_ids)
        self.positive_knowledge = [
            item for item in self.positive_knowledge if item.id in retain_ids
        ]
        self.negative_knowledge = [
            item for item in self.negative_knowledge if item.id in retain_ids
        ]

        # Modify existing knowledge
        if feedback.modify_knowledge:
            for mod in feedback.modify_knowledge:
                # Search in positive and negative knowledge
                for item in self.positive_knowledge + self.negative_knowledge:
                    if item.id == mod.knowledge_id:
                        item.content = mod.new_content
                        item.last_updated = datetime.now()
                        break

        # Remove knowledge items
        if feedback.remove_knowledge_ids:
            self.positive_knowledge = [
                item
                for item in self.positive_knowledge
                if item.id not in feedback.remove_knowledge_ids
            ]
            self.negative_knowledge = [
                item
                for item in self.negative_knowledge
                if item.id not in feedback.remove_knowledge_ids
            ]

        # Add new positive knowledge
        if feedback.add_positive_knowledge:
            for content in feedback.add_positive_knowledge:
                new_item = KnowledgeItem(
                    id=self.next_knowledge_id,
                    content=content,
                    type="positive",
                )
                self.positive_knowledge.append(new_item)
                self.next_knowledge_id += 1

        # Add new negative knowledge
        if feedback.add_negative_knowledge:
            for content in feedback.add_negative_knowledge:
                new_item = KnowledgeItem(
                    id=self.next_knowledge_id,
                    content=content,
                    type="negative",
                )
                self.negative_knowledge.append(new_item)
                self.next_knowledge_id += 1

        self._save()


def get_reasoner_ids(brain_client):
    @brain_client.reasoner(schema=OperationalResult)
    def operational_reasoner(
        input_query: str, positive_context: str, negative_context: str
    ):
        system_prompt = """You are an intelligent agent designed to provide accurate and consistent answers based on generalized knowledge.
    Apply the positive knowledge effectively and avoid using knowledge items classified as negative.
    Focus on transferring general principles to solve new queries."""

        user_prompt = f"""Query: {input_query}

    {positive_context}

    {negative_context}

    Important:
    - Use the positive knowledge items to guide your answer.
    - Avoid using any strategies or patterns mentioned in the negative knowledge items.
    - Do not memorize specific input-output pairs; instead, apply general principles.
    - Provide your answer that aligns with the main goal."""

        return system_prompt, user_prompt

    @brain_client.reasoner(schema=FeedbackResult)
    def feedback_reasoner(
        main_goal: str,
        input_query: str,
        answer: OperationalResult,
        correct_answer: str,
        positive_context: str,
        negative_context: str,
    ):
        system_prompt = f"""You are a sophisticated learning feedback agent dedicated to extracting and refining knowledge to achieve the main goal.

    MAIN GOAL: {main_goal}

    Your Objectives:
    1. Analyze the differences between the system's answer and the correct answer.
    2. Provide feedback on the system's reasoning in relation to the main goal.
    3. Extract generalized, transferable knowledge that aligns with the main goal.
    4. Identify what worked (positive knowledge) and what didn't work (negative knowledge).
    5. Suggest modifications to existing knowledge or propose new knowledge items.
    6. Enhance the system's ability to generalize to unseen data through the extracted knowledge.

    Core Principles:
    - Focus on underlying principles and patterns.
    - Separate positive insights from negative ones.
    - Ensure all knowledge is goal-oriented and transferable.
    """

        user_prompt = f"""# DEEP LEARNING ANALYSIS

    **GOAL TO ACHIEVE:** {main_goal}

    ## Current Learning Instance:
    - **Input Query:** {input_query}
    - **System Output:** {answer.answer}
    - **System Explanation:** {answer.explanation}
    - **Expected Output:** {correct_answer}

    ## Existing Knowledge:
    ### Positive Knowledge:
    {positive_context}

    ### Negative Knowledge:
    {negative_context}

    ## Your Tasks:

    ### 1. Analysis:
    - Compare the system's output with the expected output.
    - Analyze the system's reasoning in relation to the main goal.
    - Identify fundamental understandings that would help achieve the goal.
    - Reveal deeper patterns when comparing outputs.
    - Determine core principles that would improve goal achievement.

    ### 2. Feedback:
    - Provide detailed feedback on what can be learned regarding the main goal.
    - Highlight what worked (positive aspects) and what didn't work (negative aspects).
    - Focus on general principles and patterns, not specific input-output pairs.

    ### 3. Knowledge Management:
    - **Retain Knowledge**: List IDs of positive and negative knowledge items that should be retained as they are.
    - **Modify Knowledge**: Suggest modifications to existing knowledge items that need enhancement or correction. Provide in the format:
    [[knowledge_id, "enhanced or corrected content"]]
    - **Add New Knowledge**:
    - **New Positive Knowledge**: Add detailed new positive knowledge items that capture essential goal-relevant information.
    - **New Negative Knowledge**: Add detailed new negative knowledge items that capture what should be avoided.
    - **Remove Knowledge**: List IDs of knowledge items that should be removed if they are incorrect or unhelpful toward achieving the main goal.

    ## Instructions:

    - **Use the provided IDs** to refer to specific knowledge items.
    - **Do not memorize specific input-output pairs**; focus on generalizing principles to future similar queries.
    - **Make the knowledge detailed and actionable** to effectively guide the system's learning process.
    - **Add new knowledge** if existing knowledge is insufficient for future similar (not identical) queries.
    - **Remove existing knowledge** only if it is incorrect and not relevant for future similar queries.
    - **Modify existing knowledge** if it needs enhancement to achieve the main goal for future similar queries.
    - **Retain existing knowledge** if it is important for achieving the main goal in future similar queries.

    ## Focus Areas:

    - Emphasize **deep understanding** over surface patterns.
    - Prioritize **core principles** over specific solutions.
    - Ensure knowledge is **transferable** to new, unseen data.
    - Focus on **fundamental concepts** over specific applications.
    - Concentrate on mechanisms that achieve the **main goal** over specific input-output pairs.

    - If something is working, **provide positive feedback** to reinforce it.
    - If something is not working, **provide negative feedback** to suppress it.

    Important Notes:

    Ensure that all parts of the output are filled appropriately.
    Include all reasoning and analysis within the "feedback" field.
    Be concise but thorough in your feedback and knowledge suggestions.
    Do not include any extraneous text outside of the specified output format.
    Focus on extracting general principles that will help improve future performance toward achieving the main goal.
    """
        return system_prompt, user_prompt

    operational_reasoner_id = operational_reasoner.register()
    feedback_reasoner_id = feedback_reasoner.register()
    return operational_reasoner_id, feedback_reasoner_id


from dataclasses import dataclass


@dataclass
class InferenceResult:
    answer: str
    explanation: str
    positive_context: str
    negative_context: str


class CLU:
    def __init__(
        self,
        brain_client,
        main_goal="Learn to provide accurate and consistent responses",
        memory_file: str = "memory_tmp1.json",
    ):
        self.memory = Memory(file_path=memory_file)
        self.main_goal = main_goal
        self.brain_client = brain_client
        self.operational_reasoner_id, self.feedback_reasoner_id = get_reasoner_ids(
            self.brain_client
        )

    def inference(self, input_query: str):
        """
        Perform inference using the operational reasoner and return the result
        along with the positive and negative knowledge contexts.
        """
        positive_context = self.memory.get_positive_knowledge_context()
        negative_context = self.memory.get_negative_knowledge_context()

        result = self.brain_client.use(self.operational_reasoner_id)(
            input_query=input_query,
            positive_context=positive_context,
            negative_context=negative_context,
        )

        return InferenceResult(
            answer=result.answer,
            explanation=result.explanation,
            positive_context=positive_context,
            negative_context=negative_context,
        )

    def feedback(
        self,
        input_query: str,
        expected_output: str,
        operational_result: OperationalResult,
    ):
        """
        Use the feedback reasoner to compare the expected output with the
        operational result and update the memory based on the feedback.
        """
        positive_context = self.memory.get_positive_knowledge_context()
        negative_context = self.memory.get_negative_knowledge_context()

        feedback_result = self.brain_client.use(self.feedback_reasoner_id)(
            main_goal=self.main_goal,
            input_query=input_query,
            answer=operational_result,
            correct_answer=expected_output,
            positive_context=positive_context,
            negative_context=negative_context,
        )

        # Update memory with feedback
        self.memory.update_from_feedback(feedback_result)

        return feedback_result

    def process_query(self, input_query: str, correct_answer: Optional[str] = None):
        """
        Process a query, perform inference, and optionally update the memory using feedback.
        """
        # Step 1: Perform inference
        inference_result = self.inference(input_query)

        # Step 2: If correct_answer is provided, generate feedback and update memory
        if correct_answer:
            operational_result = OperationalResult(
                answer=inference_result.answer,
                explanation=inference_result.explanation,
            )
            feedback_result = self.feedback(
                input_query=input_query,
                expected_output=correct_answer,
                operational_result=operational_result,
            )
            return {
                "inference_result": inference_result,
                "feedback": feedback_result,
            }

        # Return inference result if no feedback is needed
        return inference_result
