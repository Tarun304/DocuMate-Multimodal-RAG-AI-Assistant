from typing import TypedDict, Dict, Any, List, Optional, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from typing import List
from src.backend.retriever import Retriever
from src.utils.config import config
from src.utils.logger import logger
from langsmith import traceable



# Define the workflow state
class WorkflowState(TypedDict):
    question: str # to store the working question
    docs: List[Any] # to store the retrieved context 
    context_for_ui: Dict[str, List[Any]] # Texts, Tables, Images (Retrieved)
    kind: Optional[str] # Routing field - new or follow_up
    enough: Optional[bool] # Routing field - True or False
    answer: str  # to store the final answer


# Define the Pydantic models for structured output

class QueryType(BaseModel):
    kind: Literal['new', 'follow_up'] = Field(..., description="Classification result: 'new' for standalone queries, 'follow_up' if it depends on conversation context.")


class ContextDecision(BaseModel):
    enough: bool = Field(..., description="Whether the recent context is sufficient to answer the follow-up.")
    # if enough== True, assistant_answer should contain the final answer.
    assistant_answer: Optional[str] = Field(None, description="If enough=True, provide the final assistant answer.")
    # if enough== False, rephrased_question should contain a standalone query for retrieval
    rephrased_question: Optional[str] = Field(None, description="If enough=False, rewrite the follow-up as a standalone query for retrieval.")


class RetrievalAnswer(BaseModel):
   answer: str = Field(..., description="Answer for the user's question generated from retrieved documents.")


# Orchestrator class
class QAPipeline:

    def __init__(self, top_k: int=5, thread_id: str='default', checkpointer= None):
        
        self.top_k= top_k
        self.thread_id= thread_id   # Store the thread_id
        self.retriever= Retriever()  # Create an object of the Retriever class

        self.checkpointer= checkpointer or InMemorySaver()  

        # Base LLM
        base_llm= init_chat_model(config.LLM_MODEL, model_provider=config.LLM_PROVIDER) 
        
        # Initialise the LLMs with Structured outputs
        self.classifier_llm= base_llm.with_structured_output(QueryType)
        self.ctx_decision_llm =base_llm.with_structured_output(ContextDecision)
        self.retrieval_answer_llm= base_llm.with_structured_output(RetrievalAnswer)

        logger.info(f"QA-Pipeline Initialized for thread_id: {self.thread_id}")

    
    # UTILITY FUNCTIONS: 

    @traceable(name="check_run_completion", tags=["workflow:utility"], metadata={"step": "validate_state_completion"})
    def is_completed_run(self, run_state):
        """
       Determine whether a given workflow state represents a fully completed Q&A run.

        A state is considered complete if it contains:
        - A non-empty string for 'question'
        - A non-empty string for 'answer'
        - A valid classification for 'kind' (must be either 'new' or 'follow_up')

        This is used to filter out incomplete or intermediary states from the workflow history,
        ensuring only complete Q&A steps are considered for context and memory.

        Args:
            run_state (dict): The state dictionary to validate, typically from a workflow snapshot.

        Returns:
            bool: True if the state is a fully completed Q&A run, False otherwise.
        """

        logger.debug(f"Checking state completion for question: {run_state.get('question', '')[:50]}...")
        
        has_question = (
            run_state.get("question") and 
            isinstance(run_state.get("question"), str) and 
            run_state.get("question").strip() != ""
        )
        
        has_answer = (
            run_state.get("answer") and 
            isinstance(run_state.get("answer"), str) and 
            run_state.get("answer").strip() != ""
        )
        
        has_classification = (
            run_state.get("kind") in ["new", "follow_up"]
        )
        
        is_complete = has_question and has_answer and has_classification
        
        logger.debug(f"State completion check result: {is_complete} (question: {bool(has_question)}, answer: {bool(has_answer)}, kind: {bool(has_classification)})")
        
        return is_complete

    
    @traceable(name="check_workflow_completion", tags=["workflow:utility"], metadata={"step": "validate_workflow_completion"})
    def is_workflow_completion_checkpoint(self, snapshot):
        """
       Check if a workflow snapshot represents the final completion of a Q&A workflow execution.

        Criteria for completion:
        - The snapshot metadata 'source' must be 'loop' (indicates main execution loop, not input prep).
        - The snapshot's 'next' list must be empty (no further workflow nodes scheduled, i.e., END reached).

        This utility helps eliminate duplicate or partial checkpoints (such as those created during input setup)
        and ensures downstream logic only acts on truly finished Q&A workflow runs.

        Args:
            snapshot: Workflow snapshot object with 'metadata' and 'next' attributes.

        Returns:
            bool: True if the snapshot is a workflow completion checkpoint, False otherwise.
        """
        
        metadata = snapshot.metadata
        source = metadata.get('source')
        next_nodes = snapshot.next
        
        # Only workflow completions: source='loop' AND Next=() (no more nodes)
        is_completion = source == 'loop' and len(next_nodes) == 0
        
        logger.debug(f"Checkpoint completion check: source={source}, next={next_nodes}, is_completion={is_completion}")
        
        return is_completion



    @traceable(name="node_classify_query", tags=["workflow:routing"], metadata={"step": "classify_query"})
    def classify_node(self, state:WorkflowState):
        """ Classify if the user's question as 'new' or 'follow_up' and write 'kind' to state."""

        try: 
            question = state["question"] # Store the user's question 
            config = {"configurable": {"thread_id": self.thread_id}} 

            try:
                if hasattr(self, 'compiled_workflow'):
                    history_snapshots = list(self.compiled_workflow.get_state_history(config))  # Get the state history 
                    logger.info(f"Found {len(history_snapshots)} snapshots in history")
                    
                    # Only process workflow completion checkpoints
                    recent_contexts = []
                    for snapshot in history_snapshots:
                        run_state = snapshot.values
                        
                        # Use BOTH completion check AND metadata filter
                        if (self.is_completed_run(run_state) and 
                            self.is_workflow_completion_checkpoint(snapshot)):
                            
                            recent_contexts.append(f"Q: {run_state['question']}\nA: {run_state['answer']}")
                            logger.info(f"Added workflow completion: Q={run_state['question'][:50]}...")
                            
                            # Stop after finding 2 unique completed workflows
                            if len(recent_contexts) >= 2:
                                break
                    
                    recent = "\n\n".join(recent_contexts) if recent_contexts else "None."
                else:
                    logger.warning("Compiled workflow not available")
                    recent = "None."
        
            except Exception as e:
                logger.warning(f"Could not access history: {e}")
                recent = "None."

            logger.info(f"Final recent context with {len(recent_contexts) if 'recent_contexts' in locals() else 0} unique completed workflows")

            

            # Define the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """ Classify the user's message as:
                                - 'new': standalone question  
                                - 'follow_up': depends on recent conversation

                                Return either 'new' or 'follow_up'

                                Be conservative - only classify as follow_up if clear references to prior context exist."""),

                ("human", """ Latest Message:
                            {question}

                            Recent Conversation:
                            {recent}
                        """)
            ])

            # Define the chain
            chain = prompt | self.classifier_llm 
            result = chain.invoke({"question": question, "recent": recent})

            logger.info(f"classify_node: question classified as {result.kind}")

            return {'kind': result.kind}
        
        except Exception as e:
            logger.error(f"Error in classify_node: {e}", exc_info=True)
            raise



    @traceable(name="node_context_decision", tags=["workflow:routing"], metadata={"step": "context_check_and_optional_answer"})
    def context_check_node(self, state: WorkflowState):
        """
        Check if recent context is sufficient using previous run states.
        If sufficient, generate answer directly. If not, rephrase for retrieval.
        """
        try:
                followup_question = state['question']  # Store the follow up question 
                config = {"configurable": {"thread_id": self.thread_id}}
                
                try:

                    if hasattr(self, 'compiled_workflow'):
                        history_snapshots = list(self.compiled_workflow.get_state_history(config))
                        logger.info(f"Found {len(history_snapshots)} snapshots for context check")
                        
                        # Only process workflow completion checkpoints
                        context_pairs = []
                        for snapshot in history_snapshots:
                            run_state = snapshot.values
                            
                            # Use BOTH completion check AND metadata filter
                            if (self.is_completed_run(run_state) and 
                                self.is_workflow_completion_checkpoint(snapshot)):
                                
                                context_pairs.append({
                                    "question": run_state.get("question", ""),
                                    "answer": run_state.get("answer", ""),
                                    "docs": run_state.get("docs", [])
                                })
                                logger.info(f"Added workflow completion: Q={run_state['question'][:50]}...")
                                
                                # Stop after finding 2 unique completed workflows
                                if len(context_pairs) >= 2:
                                    break
                        
                        logger.info(f"Found {len(context_pairs)} unique completed context pairs")
                        
                        # Build multimodal context blocks 
                        if context_pairs:
                            context_blocks = []
                            all_images = []
                            
                            for i, pair in enumerate(context_pairs, 1):
                                # Process docs exactly like answer_from_retrieval_node
                                texts, tables, images = [], [], []
                                for doc in pair["docs"]:
                                    kind = doc.metadata.get("kind", "text")
                                    if kind == "image":
                                        images.append(doc.page_content)
                                        all_images.append(doc.page_content)
                                    elif kind == "table":
                                        tables.append(doc.page_content)
                                    else:
                                        texts.append(doc.page_content)
                                
                                # Format context for this Q&A pair
                                text_context = "\n\n".join(texts) if texts else "No text context."
                                table_context = "\n\n".join(tables) if tables else "No table context."
                                image_info = f"{len(images)} image(s) available" if images else "No images."
                                
                                context_block = f"""
                                                ═══ Previous Conversation {i} ═══
                                                Question: {pair["question"]}
                                                Answer: {pair["answer"]}

                                                Retrieved Context:
                                                [Texts]
                                                {text_context}

                                                [Tables]
                                                {table_context}

                                                [Images]
                                                {image_info}
                                                """
                                context_blocks.append(context_block)
                            
                            full_context = "\n".join(context_blocks)
                            logger.info(f"Built full context with {len(context_pairs)} pairs")
                            
                        else:
                            full_context = "No previous conversation context available."
                            all_images = []
                            logger.info("No completed context pairs found")
                    else:
                        logger.warning("Compiled workflow not available")
                        full_context = "No previous conversation context available."
                        all_images = []
                        
                except Exception as e:
                    logger.warning(f"Could not access state history: {e}")
                    full_context = "No previous conversation context available."
                    all_images = []
                
                # Build multimodal prompt
                human_parts = [{
                    "type": "text", 
                    "text": f"""Current Follow-up Question: {followup_question}

                    Previous Context:
                    {full_context}

                    """
                 }]
                
                # Add all images from previous contexts
                for img_b64 in all_images:
                    human_parts.append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
                
                # System prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Analyze if the follow-up question can be answered using only the previous conversation context.

                                **Decision criteria:**
                                - If YES: Set enough=true and provide assistant_answer
                                - If NO: Set enough=false and provide rephrased_question as standalone query

                                **Quality requirements:**
                                - Use previous Q&As and their retrieved context
                                - Be comprehensive when answering from memory
                                - Rephrase clearly for retrieval when needed

                                Return JSON: {{"enough": boolean, "assistant_answer": "...", "rephrased_question": "..."}}"""),
                    ("human", human_parts)
                ])
                
                # Execute the chain 
                try:
                    chain = prompt | self.ctx_decision_llm
                    result = chain.invoke({})
                    
                    # Safe access to response attributes
                    enough = getattr(result, 'enough', False)
                    assistant_answer = getattr(result, 'assistant_answer', "")
                    rephrased_question = getattr(result, 'rephrased_question', "")
                    
                    logger.info(f"context_check_node: LLM returned enough={enough}")
                    
                    # If context is sufficient, return the answer 
                    if enough:
                        answer_text = assistant_answer or ""
                        logger.info("context_check_node: answered from previous context (no new retrieval).")
                        
                        # add an info message to let the user know that the bot used memory.
                        info_message="USED MEMORY"
                        
                        # Explicitly set docs to empty for memory-based answers
                        
                        return {
                            "enough": True,
                            "answer": answer_text,
                            "docs": [],  #  Explicitly set for consistency
                            "context_for_ui": {"texts": [info_message], "tables": [info_message], "images": [info_message]}
                        }
                    
                    # If context is insufficient, return the rephrased question
                    else:
                        rephrased = rephrased_question or followup_question
                        logger.info("context_check_node: insufficient context — rephrasing for retrieval.")
                        return {"enough": False, "question": rephrased}
                        
                except Exception as llm_error:
                    logger.error(f"LLM call failed in context_check_node: {llm_error}")
                    # Fallback: assume insufficient context
                    return {"enough": False, "question": followup_question}
        
        except Exception as e:
                logger.error(f"Error in context_check_node: {e}", exc_info=True)
                raise



    @traceable(name='node_retrieve', tags=['workflow: retrieval'], metadata={"step": "document_retrieval"})
    def retrieve_node(self, state: WorkflowState):

        """ Retrieve docs for the current working `question` and prepare UI buckets. """

        try:  
                # Get the question
                question= state["question"]
                # Retrieve the relevant docs
                docs= self.retriever.retrieve(question, self.top_k)

                logger.debug(f"Retrieved {len(docs)} docs")

                # Split the type of data for the UI
                buckets = {"texts": [], "tables": [], "images": []}
                for d in docs:
                    kind = d.metadata.get("kind", "text")   # stored this when indexing
                    if kind == "image":
                        buckets["images"].append(d.page_content)          # base64 string
                    elif kind == "table":
                        buckets["tables"].append(d.page_content)          
                    else:
                        buckets["texts"].append(d.page_content)

                
                return {"docs": docs, "context_for_ui": buckets}

        except Exception as e:
                logger.error(f"Error in retrieve_node: {e}", exc_info=True)
                raise



    @traceable(name="node_generate_answer_from_retrieval", tags=["workflow:generation"], metadata={"step": "generate_answer_from_retrieval"})
    def answer_from_retrieval_node(self, state: WorkflowState):

        """  
        Generate an answer using retrieved docs (for new queries or rephrased follow-ups).
        Appends the Q/A to messages and returns answer + UI context. 
        
        """

        try:
                # Store the retieved documents and the user's question
                docs= state['docs']
                question= state['question']

                # Split docs into three modality lists: texts, tables, images
                texts, tables, images= [], [], []
                for d in docs:
                    kind= d.metadata.get("kind", "text")
                    if kind=='image':
                        images.append(d.page_content)  # base-64
                    elif kind=='table':
                        tables.append(d.page_content)  
                    else:
                        texts.append(d.page_content)


                # Build the system instructions
                system_prompt="""
                    You are an expert analyst answering questions using multimodal document content (texts, tables, images).
                    
                    **IMPORTANT: Only answer questions related to the provided document content. Do not engage with inappropriate, harmful, off-topic, or unrelated questions. If a question falls outside the scope of document analysis, politely decline and redirect to document-related queries.**

                    **Core principles:**
                    - Answer only based on provided documents
                    - Be precise and factual
                    - Extract key information from all modalities when relevant  
                    - If insufficient information, clearly state limitations

                    **For different content types:**
                    - **Texts**: Extract facts, concepts, relationships
                    - **Tables**: Analyze data, numbers, trends, calculations  
                    - **Images**: Describe visuals, read charts/labels, identify patterns

                    Provide clear, structured answers that directly address the user's question. """

                # Format the text and tables as normal strings
                text_context = "\n\n".join(texts) if texts else "No text context."
                table_context = "\n\n".join(tables) if tables else "No table context."

                # Build the human message
                human_parts=[
                        {"type": "text", "text": "User Question:\n" + question + "\n\n[Texts]\n" + text_context + "\n\n[Tables]\n" + table_context}
                    ]


                # Add the image parts properly
                for img_b64 in images:
                    image_url = f"data:image/jpeg;base64,{img_b64}"
                    human_parts.append({"type": "image_url", "image_url": {"url": image_url}})
                    

                # Build the prompt template
                prompt= ChatPromptTemplate.from_messages([
                    ('system', system_prompt),
                    ('human', human_parts)
                ])

                # Define the chain
                chain= prompt | self.retrieval_answer_llm
                result= chain.invoke({})

                return {'answer': result.answer}
         
         
        except Exception as e:
                logger.error(f"Error in answer_from_retrieval_node: {e}", exc_info=True)
                raise



    # Routing functions (Required for building conditional workflows)

    # Routing function 1: 
    @traceable(name="route_from_classify", tags=["workflow:routing"], metadata={"step": "route_after_classify"})
    def route_from_classify(self, state:WorkflowState, inputs=None):
        """
        Routing after `classify_node`. Uses the `kind` field written by classify_node.
        Returns the next node name or raises on unexpected values.
        """
        logger.debug(f"route_from_classify: routing for kind= { state['kind']}")
        if state["kind"]=="new":
            return "retrieve_docs"
        elif state['kind']=="follow_up":
            return "check_context_AND_answer_or_rephrase"
        else:
             raise ValueError(f"Unexpected kind: {state['kind']}")

    # Routing function 2:
    @traceable(name="route_from_context_check", tags=["workflow:routing"], metadata={"step": "route_after_context_check"})
    def route_from_context_check(self, state:WorkflowState, inputs=None):
        """
        Routing after `context_check_node`.
        - If enough == True => workflow should END (context_check already wrote answer into state)
        - If enough == False => go to retrieval for rephrased question
        """
        logger.debug(f"route_from_context_check: routing for enough={state['enough']}" )
        if state["enough"]:
            return END  # as question already in  context_check_node
        else:
            return 'retrieve_docs'


    
    # Main workflow
    @traceable(name="qa_workflow", tags=["main", "workflow"])
    def answer_question(self, question:str):
        """ Build and execute the qa workflow:

        
        START -> classify_question
               ├─ kind='new'      -> retrieve_docs -> generate_answer -> END
               └─ kind='follow_up'-> check_context_AND_answer_or_rephrase
                                          ├─ enough=True  -> END (context_check appended answer)
                                          └─ enough=False (-> rephrase question) -> retrieve_docs -> generate_answer -> END
        """

        graph= StateGraph(WorkflowState)

        # Add the nodes
        graph.add_node("classify_question", self.classify_node)
        graph.add_node("check_context_AND_answer_or_rephrase", self.context_check_node)
        graph.add_node("retrieve_docs", self.retrieve_node)
        graph.add_node("generate_answer", self.answer_from_retrieval_node)

        # Add the edges
        graph.add_edge(START, "classify_question")
        # conditional routing after classify_node
        graph.add_conditional_edges("classify_question", self.route_from_classify)
        # conditional routing after context_check_node
        graph.add_conditional_edges("check_context_AND_answer_or_rephrase", self.route_from_context_check)
        # retrieval -> answer
        graph.add_edge("retrieve_docs", "generate_answer")
        graph.add_edge("generate_answer", END)

        # Compile the workflow
        workflow = graph.compile(checkpointer=self.checkpointer)

        # CRITICAL: Store compiled workflow so nodes can access state history
        self.compiled_workflow = workflow

        # Configuration with thread_Id
        CONFIG= {"configurable": {"thread_id": self.thread_id}}


        # Initial State
        initial_state: WorkflowState = {
            "question": question,
            "docs": [],
            "context_for_ui": {"texts": [], "tables": [], "images": []},
            "kind": None,
            "enough": None,
            "answer": None,
             
        }


        # Run the workflow
        final_state= workflow.invoke(initial_state, config=CONFIG)

        # return the results
        return {

            "answer": final_state["answer"],
            "context_for_ui": final_state["context_for_ui"]


        }

    


