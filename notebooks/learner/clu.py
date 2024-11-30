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
        system_prompt = """You are an intelligent agent designed to provide accurate and consistent answers based on specific, actionable knowledge.

    Guidelines:
    - Apply the positive knowledge effectively.
    - Avoid patterns or rules mentioned in the negative knowledge.
    - Focus on generalizing patterns to solve new queries.
    - Do not memorize specific input-output pairs."""

        user_prompt = f"""Query: {input_query}

    ### Positive Knowledge (apply these patterns or rules):
    {positive_context}

    ### Negative Knowledge (avoid these patterns or rules):
    {negative_context}

    Important:
    - Use the positive knowledge items to guide your answer.
    - Avoid any strategies or patterns mentioned in the negative knowledge items.
    - Provide your answer that aligns with the main goal.
    - Think step by step before generating the output."""

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
        system_prompt = f"""You are a learning feedback agent dedicated to improving the system's performance by extracting specific, actionable knowledge from input-output pairs.

    MAIN GOAL: {main_goal}

    Your Objectives:
    1. Analyze the differences between the system's answer and the correct answer.
    2. Identify specific patterns, rules, or transformations that can explain the correct outputs.
    3. Extract actionable knowledge that can help the system achieve the main goal.
    4. Suggest modifications to existing knowledge or propose new knowledge items.
    5. Enhance the system's ability to generalize to unseen data through the extracted knowledge.

    Guidelines:
    - Focus on specific patterns or rules observed in the input-output pairs.
    - Express knowledge items clearly and specifically, avoiding vague or generic statements.
    - Ensure knowledge items are generalizable and can be applied to future queries.
    - Avoid memorizing specific input-output pairs; extract underlying principles instead.
    """

        user_prompt = f"""# LEARNING ANALYSIS

    **GOAL TO ACHIEVE:** {main_goal}

    ## Current Instance:
    - **Input Query:** {input_query}
    - **System Output:** {answer.answer}
    - **System Explanation:** {answer.explanation}
    - **Expected Output:** {correct_answer}

    ## Existing Knowledge:
    ### Positive Knowledge (IDs provided):
    {positive_context}

    ### Negative Knowledge (IDs provided):
    {negative_context}

    ## Your Tasks:

    1. **Analysis:**
    - Compare the system's output with the expected output.
    - Identify specific patterns or rules that can explain the discrepancy.
    - Determine what knowledge or reasoning is missing or incorrect.

    2. **Feedback:**
    - Provide detailed feedback on how the system can improve.
    - Highlight effective reasoning (to reinforce) and ineffective reasoning (to adjust).

    3. **Knowledge Management:**
    - **Retain Knowledge IDs:** List IDs of knowledge items to retain.
    - **Modify Knowledge:** Suggest improvements to existing knowledge items in the format:
        - `[knowledge_id, "updated content"]`
    - **Add New Knowledge:**
        - **Positive Knowledge:** New patterns or rules that help achieve the main goal.
        - **Negative Knowledge:** Patterns or reasoning to avoid.
    - **Remove Knowledge IDs:** List IDs of knowledge items to remove if they are incorrect or unhelpful.

    ## Instructions:

    - **Use IDs** when referring to knowledge items.
    - **Express knowledge items as specific patterns or rules** applicable to future queries.
    - **Avoid referencing specific input-output pairs** in the knowledge items.
    - **Ensure knowledge is actionable and directly improves performance** toward the main goal.
    ## Important:

    - Be detailed on your Knowledge accumulation.
    - Do not include any hallucinations or irrelevant information.
    - Focus on extracting knowledge that will help improve future performance in similar scenarios.

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
