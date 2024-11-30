import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


# Define KnowledgeItem model
class KnowledgeItem(BaseModel):
    id: int = Field(..., description="Unique identifier for the knowledge item.")
    content: str = Field(..., description="The content of the knowledge item.")
    knowledge_type: str = Field(
        ...,
        description="Type of knowledge, such as positive, negative, rule, pattern, etc.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp."
    )
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp."
    )


# Define NewKnowledgeItem model
class NewKnowledgeItem(BaseModel):
    content: str = Field(..., description="The content of the new knowledge item.")
    knowledge_type: str = Field(
        ...,
        description="Type of knowledge, such as positive, negative, rule, pattern, etc. these are just examples of knowledge types, but they could be anything based on the context.",
    )


# Define OperationalResult model
class OperationalResult(BaseModel):
    answer: str = Field(
        ..., description="The answer to the input query based on current knowledge."
    )
    explanation: str = Field(..., description="The reasoning behind the answer.")


# Define RuleModification model
class RuleModification(BaseModel):
    knowledge_id: int = Field(..., description="The knowledge item ID to modify.")
    new_content: str = Field(..., description="The new content for the knowledge item.")


# Define FeedbackResult model
class FeedbackResult(BaseModel):
    feedback: str = Field(..., description="Detailed feedback on the previous answer.")
    retain_knowledge_ids: List[int] = Field(
        default_factory=list,
        description="IDs of knowledge items to retain as they are.",
    )
    modify_knowledge: List[RuleModification] = Field(
        default_factory=list, description="List of knowledge items to modify."
    )
    add_knowledge: List[NewKnowledgeItem] = Field(
        default_factory=list, description="New knowledge items to add."
    )
    remove_knowledge_ids: List[int] = Field(
        default_factory=list, description="IDs of knowledge items to remove."
    )


# Define MetaFeedbackResult model
class MetaFeedbackResult(BaseModel):
    focus_areas: List[str] = Field(
        ..., description="List of focus areas or guidelines for the feedback reasoner."
    )


# Memory class to manage knowledge items
class Memory:
    def __init__(self, file_path: str = "memory.json"):
        self.file_path = Path(file_path)
        self.knowledge_items: List[KnowledgeItem] = []
        self.next_knowledge_id = 1
        self._load()

    def _load(self):
        if not self.file_path.exists():
            self._save()
            return

        data = json.loads(self.file_path.read_text())
        self.next_knowledge_id = data.get("next_knowledge_id", 1)
        self.knowledge_items = [
            KnowledgeItem(**item) for item in data.get("knowledge_items", [])
        ]

    def _save(self):
        data = {
            "next_knowledge_id": self.next_knowledge_id,
            "knowledge_items": [item.dict() for item in self.knowledge_items],
        }
        self.file_path.write_text(json.dumps(data, default=str, indent=2))

    def get_knowledge_context(self) -> str:
        if not self.knowledge_items:
            return "<KnowledgeItems></KnowledgeItems>"
        context = "<KnowledgeItems>\n"
        for item in self.knowledge_items:
            context += f'  <KnowledgeItem id="{item.id}" knowledge_type="{item.knowledge_type}">\n'
            context += f"    {item.content}\n"
            context += f"  </KnowledgeItem>\n"
        context += "</KnowledgeItems>"
        return context

    def update_from_feedback(self, feedback: FeedbackResult):
        # Retain specified knowledge items
        if feedback.retain_knowledge_ids:
            retain_ids = set(feedback.retain_knowledge_ids)
            self.knowledge_items = [
                item for item in self.knowledge_items if item.id in retain_ids
            ]

        # Modify existing knowledge
        if feedback.modify_knowledge:
            for mod in feedback.modify_knowledge:
                for item in self.knowledge_items:
                    if item.id == mod.knowledge_id:
                        item.content = mod.new_content
                        item.last_updated = datetime.now()
                        break

        # Remove knowledge items
        if feedback.remove_knowledge_ids:
            self.knowledge_items = [
                item
                for item in self.knowledge_items
                if item.id not in feedback.remove_knowledge_ids
            ]

        # Add new knowledge items
        if feedback.add_knowledge:
            for new_item in feedback.add_knowledge:
                knowledge_item = KnowledgeItem(
                    id=self.next_knowledge_id,
                    content=new_item.content,
                    knowledge_type=new_item.knowledge_type,
                )
                self.knowledge_items.append(knowledge_item)
                self.next_knowledge_id += 1

        self._save()


# Define InferenceResult dataclass
from dataclasses import dataclass


@dataclass
class InferenceResult:
    answer: str
    explanation: str
    knowledge_context: str


# Function to get reasoner IDs
def get_reasoner_ids(brain_client):
    @brain_client.reasoner(schema=OperationalResult)
    def operational_reasoner(
        input_query: str,
        knowledge_context: str,
        main_goal: str,
    ):
        system_prompt = f"""You are an intelligent agent designed to provide accurate and consistent answers based on the knowledge provided.

MAIN GOAL: {main_goal}

Guidelines:
- Use the knowledge items provided to guide your answer.
- Pay attention to the type of each knowledge item (e.g., positive, negative, rule, pattern, etc.). These are examples of knowledge types, but they could be anything based on the context.
- Apply the knowledge appropriately to generate your answer.
- Do not memorize specific input-output pairs."""

        user_prompt = f"""## Query:
{input_query}

### Knowledge Items:
{knowledge_context}

Important:
- Use the knowledge items to guide your answer.
- Provide your answer that aligns with the main goal.
- Think step by step before generating the output."""

        return system_prompt, user_prompt

    @brain_client.reasoner(schema=MetaFeedbackResult)
    def meta_feedback_reasoner(
        main_goal: str,
        input_query: str,
        system_output: str,
        expected_output: str,
    ):
        system_prompt = f"""You are a meta-feedback agent whose role is to determine what the feedback agent should focus on to improve the system's performance.

MAIN GOAL: {main_goal}

Your Objectives:
1. Analyze the differences between the system's output and the expected output.
2. Determine specific areas, topics, or aspects that the feedback agent should focus on to help the system learn and improve.
3. Provide a list of focus areas or guidelines for the feedback agent.

Consider what knowledge gaps exist and what learning would most effectively improve future performance."""

        user_prompt = f"""## Analysis Context:

- **Input Query:** {input_query}
- **System Output:** {system_output}
- **Expected Output:** {expected_output}

## Instructions:

- Identify key areas where the system needs improvement.
- Determine what the feedback agent should focus on when providing feedback.
- Focus on areas that will help the system generalize to new, unseen data.
- Provide a concise list of focus areas or guidelines.
- Guide if it needs to learn a non working method or rule or pattern or knowdlge type, or if something worked let it know that it worked and ask it to understand why it worked and store it in knowledge base for future reference.
- This is a meta feedback that will guide feedback reasoner, which intern will work almost like reinforcement learning feedback loop for us to extract knowledge from the system guided by main goal.

## Output Format:

Provide your response in the following format:

- **Focus Areas:**
  - "First focus area"
  - "Second focus area"
  - ...

Do not include any additional text outside of the focus areas list."""

        return system_prompt, user_prompt

    @brain_client.reasoner(schema=FeedbackResult)
    def feedback_reasoner(
        main_goal: str,
        input_query: str,
        answer: OperationalResult,
        correct_answer: str,
        knowledge_context: str,
        focus_areas: List[str],
    ):
        system_prompt = f"""You are a learning feedback agent dedicated to improving the system's performance by extracting specific, actionable knowledge from input-output pairs.

MAIN GOAL: {main_goal}

Your Objectives:
1. Analyze the differences between the system's answer and the correct answer.
2. Based on the focus areas provided, decide what types of knowledge items are needed (e.g., positive, negative, rule, pattern, etc.), these are just examples of knowledge types, but they could be anything based on the context.
3. Provide feedback and suggest knowledge items to add, modify, or remove.
4. Enhance the system's ability to generalize to unseen data through the extracted knowledge.

Guidelines:
- Focus on the specific areas mentioned in the focus areas.
- Express knowledge items clearly and specifically, avoiding vague or generic statements.
- Ensure knowledge items are generalizable and can be applied to future queries.
- Avoid memorizing specific input-output pairs; extract underlying principles instead.
"""

        focus_areas_formatted = "\n".join(f"- {area}" for area in focus_areas)

        user_prompt = f"""# LEARNING ANALYSIS

**GOAL TO ACHIEVE:** {main_goal}

## Current Instance:
- **Input Query:** {input_query}
- **System Output:** {answer.answer}
- **System Explanation:** {answer.explanation}
- **Expected Output:** {correct_answer}

## Focus Areas:
{focus_areas_formatted}

## Existing Knowledge Items:
{knowledge_context}

## Your Tasks:

1. **Analysis:**
- Based on the focus areas, compare the system's output with the expected output.
- Identify specific type of knowldge we can extract that can explain the discrepancy.
- Determine what knowledge or reasoning is missing or incorrect.

2. **Knowledge Management:**
- **Retain Knowledge IDs:** List IDs of knowledge items to retain.
- **Modify Knowledge:** Suggest improvements to existing knowledge items in the format:
    - [knowledge_id, "updated content"]
- **Add New Knowledge:** Provide new knowledge items to add, specifying the knowledge_type and content. Use the format:
    - [knowledge_type, "content"]
- **Remove Knowledge IDs:** List IDs of knowledge items to remove if they are incorrect or unhelpful.

## Instructions:

- **Use IDs** when referring to knowledge items.
- **Express knowledge items as specific patterns or rules** applicable to future queries.
- **Ensure knowledge is actionable and directly improves performance** toward the main goal.
- **Do not include any extraneous text** outside of the specified output format.

## Important:

- Be detailed in your knowledge accumulation.
- Do not include any hallucinations or irrelevant information.
- Focus on extracting knowledge that will help improve future performance in similar scenarios.

"""

        return system_prompt, user_prompt

    operational_reasoner_id = operational_reasoner.register()
    meta_feedback_reasoner_id = meta_feedback_reasoner.register()
    feedback_reasoner_id = feedback_reasoner.register()
    return operational_reasoner_id, meta_feedback_reasoner_id, feedback_reasoner_id


# CLU class to manage the overall process
class CLU:
    def __init__(
        self,
        brain_client,
        main_goal="Learn to provide accurate and consistent responses",
        memory_file: str = "memory.json",
    ):
        self.memory = Memory(file_path=memory_file)
        self.main_goal = main_goal
        self.brain_client = brain_client
        (
            self.operational_reasoner_id,
            self.meta_feedback_reasoner_id,
            self.feedback_reasoner_id,
        ) = get_reasoner_ids(self.brain_client)

    def inference(self, input_query: str):
        """
        Perform inference using the operational reasoner and return the result
        along with the knowledge context.
        """
        knowledge_context = self.memory.get_knowledge_context()

        result = self.brain_client.use(self.operational_reasoner_id)(
            input_query=input_query,
            knowledge_context=knowledge_context,
            main_goal=self.main_goal,
        )

        return InferenceResult(
            answer=result.answer,
            explanation=result.explanation,
            knowledge_context=knowledge_context,
        )

    def meta_feedback(
        self,
        input_query: str,
        system_output: str,
        expected_output: str,
    ) -> MetaFeedbackResult:
        """
        Use the meta-feedback reasoner to determine focus areas for the feedback reasoner.
        """
        meta_feedback_result = self.brain_client.use(self.meta_feedback_reasoner_id)(
            main_goal=self.main_goal,
            input_query=input_query,
            system_output=system_output,
            expected_output=expected_output,
        )
        return meta_feedback_result

    def feedback(
        self,
        input_query: str,
        expected_output: str,
        operational_result: OperationalResult,
        focus_areas: List[str],
    ):
        """
        Use the feedback reasoner to compare the expected output with the
        operational result and update the memory based on the feedback.
        """
        knowledge_context = self.memory.get_knowledge_context()

        feedback_result = self.brain_client.use(self.feedback_reasoner_id)(
            main_goal=self.main_goal,
            input_query=input_query,
            answer=operational_result,
            correct_answer=expected_output,
            knowledge_context=knowledge_context,
            focus_areas=focus_areas,
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
            # Prepare operational result
            operational_result = OperationalResult(
                answer=inference_result.answer,
                explanation=inference_result.explanation,
            )
            # Step 2a: Get meta-feedback to determine focus areas
            meta_feedback_result = self.meta_feedback(
                input_query=input_query,
                system_output=operational_result.answer,
                expected_output=correct_answer,
            )
            # Step 2b: Use feedback reasoner with focus areas
            feedback_result = self.feedback(
                input_query=input_query,
                expected_output=correct_answer,
                operational_result=operational_result,
                focus_areas=meta_feedback_result.focus_areas,
            )
            return {
                "inference_result": inference_result,
                "meta_feedback": meta_feedback_result,
                "feedback": feedback_result,
            }

        # Return inference result if no feedback is needed
        return {"inference_result": inference_result}
