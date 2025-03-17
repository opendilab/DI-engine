from typing import Tuple, Optional, List, Dict
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import re
import math
import json
from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel


@REWARD_MODEL_REGISTRY.register("math_rule")
class MathRuleRewardModel(BaseRewardModel):
    """
    Math rule-based reward model for evaluating mathematical answers.
    Supports various mathematical expression formats including LaTeX, fractions, percentages, etc.
    """

    config = dict(
        # (str) The type of the reward model.
        type="math_rule",
        # (str) The name of the dataset, usually the huggingface dataset name.
        dataset_name="",
        # (str) The name of the tokenizer, usually the huggingface tokenizer name.
        tokenizer_name="",
        # (float) The score of format error.
        format_error_reward=-2,
        # (float) The score of answer error.
        answer_error_reward=-1,
        # (float) The score of correct.
        correct_reward=1,
        # (float) Relative tolerance for numerical comparison
        rel_tol=1e-5,
        # (float) Absolute tolerance for numerical comparison
        abs_tol=1e-8,
    )

    def __init__(
            self,
            config: EasyDict,
            device: str = "cpu",
            logger=None,
            tb_logger: "SummaryWriter" = None,
    ) -> None:  # noqa
        """Initialize the math rule reward model"""
        self.cfg = config
        self.device = device
        self.logger = logger
        self.tb_logger = tb_logger

        # Initialize tokenizer
        if hasattr(config, "tokenizer_name") and config.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            self.pad_token = (self.tokenizer.pad_token if self.tokenizer.pad_token else "[PAD]")
            self.eos_token = (self.tokenizer.eos_token if self.tokenizer.eos_token else "[EOS]")
        else:
            self.tokenizer = None
            self.pad_token = "[PAD]"
            self.eos_token = "[EOS]"

    def _process_target_answer(self, text: str) -> Optional[float]:
        """Process target answer text and convert to numerical value"""
        if text is None or not text.strip():
            return None
        # Clean and normalize text
        if self.tokenizer:
            text = strip_sequence(text, self.pad_token, self.eos_token)
        text = normalize_text(text)

        # Try to process the mathematical expression
        try:
            result = self._process_math_expression(text)
            if result is not None:
                return result
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error processing target answer: {e}")
            return None

    def _process_response_answer(self, response: str) -> Tuple[Optional[float], Optional[str]]:
        """Process response text, extract and convert to numerical value"""
        if response is None or not response.strip():
            return None, None

        # Clean text
        if self.tokenizer:
            response = strip_sequence(response, self.pad_token, self.eos_token)

        # First try to extract the final answer
        final_answer = self._extract_final_answer(response)

        # If a final answer is extracted, try to process it
        if final_answer:
            try:
                value = self._process_math_expression(final_answer)
                if value is not None:
                    return value, final_answer
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"Error processing final answer: {e}")

        # If unable to get a valid value from the final answer, try to extract all possible expressions
        expressions = self._extract_all_expressions(response)

        # Try to process each expression until a valid answer is found
        for expr in expressions:
            try:
                value = self._process_math_expression(expr)
                if value is not None:
                    return value, expr
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"Error processing expression '{expr}': {e}")

        # If all attempts fail, return None
        return None, None

    def _check_answer_match(self, pred: Optional[float], target: Optional[float]) -> bool:
        """Check if two answers match within tolerance"""
        if pred is None or target is None:
            return False
        try:
            return math.isclose(
                pred,
                target,
                rel_tol=self.cfg.get("rel_tol", 1e-5),
                abs_tol=self.cfg.get("abs_tol", 1e-8),
            )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error comparing answers: {e}")
            return False

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """
        Extract the final answer from text.
        Supports various formats:
        1. "The answer is X"
        2. "Therefore, X is the answer"
        3. "X" (if only one number)
        4. "\\boxed{X}"
        5. "= X" (expression after equals sign)
        6. Last LaTeX expression like \\frac{a}{b}, \\sqrt{x}, etc.
        """
        # Try to extract boxed content
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            return boxed_match.group(0)

        # Try to extract "the answer is X" format
        answer_match = re.search(r"(?:the\s+)?answer\s+is\s+([^\.]+)", text, re.IGNORECASE)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            # Check if the extracted answer contains a LaTeX expression
            latex_match = re.search(r"(\\frac\{[^}]+\}\{[^}]+\}|\\sqrt\{[^}]+\})", answer_text)
            if latex_match:
                return latex_match.group(0)
            return answer_text

        # Try to extract "therefore, X is the answer" format
        therefore_match = re.search(r"therefore,?\s+([^\.]+)\s+is\s+the\s+answer", text, re.IGNORECASE)
        if therefore_match:
            therefore_text = therefore_match.group(1).strip()
            # Check if the extracted answer contains a LaTeX expression
            latex_match = re.search(r"(\\frac\{[^}]+\}\{[^}]+\}|\\sqrt\{[^}]+\})", therefore_text)
            if latex_match:
                return latex_match.group(0)
            return therefore_text

        # Try to extract expression after equals sign
        equals_matches = re.findall(r"=\s*([^\.=]+?)(?:\.|$|=)", text)
        if equals_matches:
            last_eq = equals_matches[-1].strip()
            # Check if there's a LaTeX expression after the equals sign
            latex_match = re.search(r"(\\frac\{[^}]+\}\{[^}]+\}|\\sqrt\{[^}]+\})", last_eq)
            if latex_match:
                return latex_match.group(0)
            return last_eq

        # Try to directly extract LaTeX fraction expression
        frac_matches = re.findall(r"(\\frac\{[^}]+\}\{[^}]+\})", text)
        if frac_matches:
            return frac_matches[-1]

        # Try to directly extract LaTeX square root expression
        sqrt_matches = re.findall(r"(\\sqrt\{[^}]+\})", text)
        if sqrt_matches:
            return sqrt_matches[-1]

        # Try to extract pi-related expressions
        pi_expr = self._extract_pi_expressions(text)
        if pi_expr:
            return pi_expr

        # If there's only one number, return it directly
        numbers = re.findall(r"-?\d*\.?\d+", text)
        if len(numbers) == 1:
            return numbers[0]

        # Try to extract the last number (as a fallback)
        if numbers:
            return numbers[-1]

        return None

    def _extract_pi_expressions(self, text: str) -> Optional[str]:
        """Extract pi-related expressions from text"""
        # Try to extract expressions like \frac{a\pi}{b}
        pi_frac_matches = re.findall(r"(\\frac\{[^}]*\\pi[^}]*\}\{[^}]+\})", text)
        if pi_frac_matches:
            return pi_frac_matches[-1]

        # Try to extract expressions like \frac{a}{b}\pi
        frac_pi_matches = re.findall(r"(\\frac\{[^}]+\}\{[^}]+\}\\pi)", text)
        if frac_pi_matches:
            return frac_pi_matches[-1]

        # Try to extract expressions like 11π/6
        text_with_pi = text.replace("\\pi", "π")
        pi_div_matches = re.findall(r"(\d+π/\d+)", text_with_pi)
        if pi_div_matches:
            return pi_div_matches[-1]

        # Try to extract expressions like π/2
        pi_simple_div_matches = re.findall(r"(π/\d+)", text_with_pi)
        if pi_simple_div_matches:
            return pi_simple_div_matches[-1]

        # Try to extract expressions like 2π
        pi_mult_matches = re.findall(r"(\d+π)", text_with_pi)
        if pi_mult_matches:
            return pi_mult_matches[-1]

        # Check for standalone π
        if "π" in text_with_pi or "\\pi" in text:
            pi_standalone = re.search(r"(^|[^a-zA-Z0-9])π($|[^a-zA-Z0-9])", text_with_pi)
            if pi_standalone:
                return "π"

        return None

    def _process_pi_expressions(self, text: str) -> Optional[float]:
        """Process pi-related expressions and convert to numerical value"""
        # Standardize pi notation
        text = text.replace("\\pi", "π")

        # Process expressions like 11π/6
        pi_match = re.search(r"(\d+)π/(\d+)", text)
        if pi_match:
            num, denom = map(int, pi_match.groups())
            return (num * math.pi) / denom

        # Process expressions like π/2
        pi_div_match = re.search(r"π/(\d+)", text)
        if pi_div_match:
            denom = int(pi_div_match.group(1))
            return math.pi / denom

        # Process expressions like 2π
        pi_mult_match = re.search(r"(\d+)π", text)
        if pi_mult_match:
            num = int(pi_mult_match.group(1))
            return num * math.pi

        # If just π
        if text == "π":
            return math.pi

        return None

    def _process_math_expression(self, text: str) -> Optional[float]:
        """
        Process special mathematical expressions, such as:
        1. Fractions: 1/2, \\frac{1}{2}
        2. Percentages: 50%
        3. Scientific notation: 1.2e-3
        4. Mixed expressions: 1 + 2/3
        5. Square roots: \\sqrt{2}
        6. Mixed fractions: 1\\frac{1}{2}
        7. Max/min functions: \\max(1,2,3), \\min(1,2,3)
        8. Pi-related expressions: 11π/6, \\frac{11\\pi}{6}
        """
        if text is None or not text.strip():
            return None

        try:
            # Remove all spaces and unnecessary LaTeX commands
            text = text.replace(" ", "")
            text = text.replace("\\left", "").replace("\\right", "")

            # Process pi-related expressions
            if "π" in text or "\\pi" in text:
                pi_value = self._process_pi_expressions(text)
                if pi_value is not None:
                    return pi_value

            # Process percentages
            if "%" in text:
                return float(text.replace("%", "")) / 100

            # Process LaTeX square roots \sqrt{...}
            sqrt_match = re.search(r"\\sqrt\{([^}]+)\}", text)
            if sqrt_match:
                inner_expr = sqrt_match.group(1)
                inner_value = self._process_math_expression(inner_expr)
                if inner_value is not None:
                    return math.sqrt(inner_value)

            # Process LaTeX fractions \frac{...}{...}
            frac_match = re.search(r"\\frac\{([^}]+)\}\{([^}]+)\}", text)
            if frac_match:
                num = frac_match.group(1)
                denom = frac_match.group(2)

                # Recursively process numerator and denominator
                num_value = self._process_math_expression(num)
                denom_value = self._process_math_expression(denom)

                if (num_value is not None and denom_value is not None and denom_value != 0):
                    return num_value / denom_value

            # Process mixed fractions 1\frac{1}{2}
            mixed_frac_match = re.search(r"(\d+)\\frac\{([^}]+)\}\{([^}]+)\}", text)
            if mixed_frac_match:
                whole = int(mixed_frac_match.group(1))
                num = mixed_frac_match.group(2)
                denom = mixed_frac_match.group(3)

                # Recursively process numerator and denominator
                num_value = self._process_math_expression(num)
                denom_value = self._process_math_expression(denom)

                if (num_value is not None and denom_value is not None and denom_value != 0):
                    return whole + (num_value / denom_value)

            # Process max function \max(a,b,c)
            max_match = re.search(r"\\max\(([^)]+)\)", text)
            if max_match:
                values_str = max_match.group(1)
                values = values_str.split(",")
                processed_values = []
                for val in values:
                    processed_val = self._process_math_expression(val)
                    if processed_val is not None:
                        processed_values.append(processed_val)
                if processed_values:
                    return max(processed_values)

            # Process min function \min(a,b,c)
            min_match = re.search(r"\\min\(([^)]+)\)", text)
            if min_match:
                values_str = min_match.group(1)
                values = values_str.split(",")
                processed_values = []
                for val in values:
                    processed_val = self._process_math_expression(val)
                    if processed_val is not None:
                        processed_values.append(processed_val)
                if processed_values:
                    return min(processed_values)

            # Process simple arithmetic operations
            if any(op in text for op in ["+", "-", "*", "/"]):
                # Safe eval, only allowing basic operations
                safe_dict = {"__builtins__": None}
                return float(eval(text, safe_dict))

            # Process scientific notation
            if "e" in text.lower() and re.match(r"-?\d+\.?\d*e[+-]?\d+", text.lower()):
                return float(text)

            # Process regular numbers
            return float(text)
        except Exception as e:
            # Log exception information for debugging
            if self.logger:
                self.logger.debug(f"Error processing math expression '{text}': {str(e)}")
            return None

    def estimate(self, data: List[Dict]) -> List[Dict]:
        """
        Overview:
            Estimate rewards for mathematical answers based on rule-based comparison.
        Arguments:
            - data (:obj:`List[Dict]`): The list of data queries used for estimation.
                Format: [{"question": "...", "answer": "...", "response": "..."}, ...]
                Each dictionary may contain:
                - question: The mathematical question
                - answer: The ground truth answer
                - response: The model's response to evaluate
                - system: Optional system prompt
                - query: Optional alternative to question
        Returns:
            - rewards (:obj:`List[Dict]`): The estimated rewards.
                Each dictionary contains:
                - reward: The numerical reward value
                - metadata: Additional information about the evaluation
        Examples:
            >>> data = [{
            >>>     "question": "What is 2+2?",
            >>>     "answer": "4",
            >>>     "response": "The answer is 4."
            >>> }]
            >>> results = model.estimate(data)
            >>> print(results[0]["reward"])  # 1.0 (correct)
            >>> print(results[0]["metadata"]["reason"])  # "correct_answer"
        """
        rewards = []

        for item in data:
            result = {
                "reward": self.cfg.format_error_reward,
                "metadata": {
                    "reason": "format_error",
                    "response_value": None,
                    "target_value": None,
                    "match_result": False,
                    "extracted_code": None,
                    "final_answer": None,
                    "extracted_expressions": [],
                },
            }

            try:
                # Extract question, answer and response from data item
                item_data = self._extract_item_data(item)
                if item_data is None:
                    rewards.append(result)
                    continue

                question, gt_answer, response = item_data

                # Process target answer
                target_value = self._process_target_answer(gt_answer)
                result["metadata"]["target_value"] = target_value

                # Process response answer
                response_value, final_answer = self._process_response_answer(response)
                result["metadata"]["response_value"] = response_value
                result["metadata"]["final_answer"] = final_answer

                # Extract Python code (if any)
                extracted_code = self._extract_python_code(response)
                result["metadata"]["extracted_code"] = extracted_code

                # Extract all possible expressions (for debugging)
                expressions = self._extract_all_expressions(response)
                result["metadata"]["extracted_expressions"] = expressions

                # Determine reward based on answer comparison
                result = self._determine_reward(result, target_value, response_value)

            except Exception as e:
                result["metadata"]["reason"] = f"error: {str(e)}"
                if self.logger:
                    self.logger.error(f"Error evaluating data: {str(e)}")

            rewards.append(result)

        return rewards

    def _extract_item_data(self, item) -> Optional[Tuple[str, str, str]]:
        """Extract question, answer and response from data item"""
        if isinstance(item, dict):
            question = item.get("question", "")
            gt_answer = item.get("answer", "")
            response = item.get("response", "")
            query = item.get("query", "")
        elif isinstance(item, str):
            # If input is a string, try to parse as JSON
            try:
                item_dict = json.loads(item)
                question = item_dict.get("question", "")
                gt_answer = item_dict.get("answer", "")
                response = item_dict.get("response", "")
                query = item_dict.get("query", "")
            except:
                # If parsing fails, assume the entire string is the response
                question = ""
                gt_answer = ""
                response = item
                query = ""
        else:
            # Unsupported input type
            return None

        # If no question but query exists, use query as question
        if not question and query:
            question = query

        return question, gt_answer, response

    def _determine_reward(
            self,
            result: Dict,
            target_value: Optional[float],
            response_value: Optional[float],
    ) -> Dict:
        """Determine reward based on answer comparison"""
        if target_value is None:
            result["reward"] = self.cfg.format_error_reward
            result["metadata"]["reason"] = "invalid_target_format"
        elif response_value is None:
            result["reward"] = self.cfg.format_error_reward
            result["metadata"]["reason"] = "invalid_response_format"
        else:
            # Compare answers
            is_match = self._check_answer_match(response_value, target_value)
            result["metadata"]["match_result"] = is_match

            if is_match:
                result["reward"] = self.cfg.correct_reward
                result["metadata"]["reason"] = "correct_answer"
            else:
                result["reward"] = self.cfg.answer_error_reward
                result["metadata"]["reason"] = "wrong_answer"

        return result

    def _extract_all_expressions(self, text: str) -> List[str]:
        """Extract all possible mathematical expressions from text, sorted by priority"""
        if text is None or not text.strip():
            return []

        expressions = []

        # Extract expressions from LaTeX math environments
        self._extract_latex_environments(text, expressions)

        # Extract boxed content (highest priority)
        self._extract_boxed_content(text, expressions)

        # Extract expressions after equals sign
        self._extract_equals_expressions(text, expressions)

        # Extract expressions from answer phrases
        self._extract_answer_phrases(text, expressions)

        # Extract LaTeX expressions
        self._extract_latex_expressions(text, expressions)

        # Extract pi-related expressions
        self._extract_pi_expressions_for_list(text, expressions)

        # Extract all numbers (lowest priority)
        self._extract_numbers(text, expressions)

        # Remove duplicates while preserving order
        unique_expressions = []
        for expr in expressions:
            if expr not in unique_expressions:
                unique_expressions.append(expr)

        return unique_expressions

    def _extract_latex_environments(self, text: str, expressions: List[str]) -> None:
        """Extract expressions from LaTeX math environments"""
        # Match \(...\) or $...$ format LaTeX expressions
        latex_envs = re.findall(r"\\\\?\((.+?)\\\\?\)", text) + re.findall(r"\$(.+?)\$", text)
        for latex_env in latex_envs:
            expressions.append(latex_env.strip())

    def _extract_boxed_content(self, text: str, expressions: List[str]) -> None:
        """Extract boxed content"""
        boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
        for match in boxed_matches:
            expressions.append(match.strip())

    def _extract_equals_expressions(self, text: str, expressions: List[str]) -> None:
        """Extract expressions after equals sign"""
        equals_matches = re.findall(r"=\s*([^\.=]+?)(?:\.|$|=)", text)
        for match in equals_matches:
            expressions.append(match.strip())

    def _extract_answer_phrases(self, text: str, expressions: List[str]) -> None:
        """Extract expressions from answer phrases"""
        # Extract "the answer is X" format
        answer_match = re.search(r"(?:the\s+)?answer\s+is\s+([^\.]+)", text, re.IGNORECASE)
        if answer_match:
            expressions.append(answer_match.group(1).strip())

        # Extract "therefore, X is the answer" format
        therefore_match = re.search(r"therefore,?\s+([^\.]+)\s+is\s+the\s+answer", text, re.IGNORECASE)
        if therefore_match:
            expressions.append(therefore_match.group(1).strip())

    def _extract_latex_expressions(self, text: str, expressions: List[str]) -> None:
        """Extract LaTeX expressions"""
        # Extract LaTeX fraction expressions
        frac_matches = re.findall(r"\\frac\{([^}]+)\}\{([^}]+)\}", text)
        for num, denom in frac_matches:
            expressions.append(f"\\frac{{{num}}}{{{denom}}}")

        # Extract LaTeX square root expressions
        sqrt_matches = re.findall(r"\\sqrt\{([^}]+)\}", text)
        for inner in sqrt_matches:
            expressions.append(f"\\sqrt{{{inner}}}")

        # Extract all LaTeX expressions
        latex_expressions = re.findall(r"\\[a-zA-Z]+(?:\{[^}]*\})+", text)
        for expr in latex_expressions:
            if expr not in expressions:
                expressions.append(expr)

    def _extract_pi_expressions_for_list(self, text: str, expressions: List[str]) -> None:
        """Extract pi-related expressions for the expressions list"""
        # Replace \pi with π for unified processing
        text_with_pi = text.replace("\\pi", "π")

        # Extract expressions like 11π/6
        pi_div_matches = re.findall(r"(\d+)π/(\d+)", text_with_pi)
        for num, denom in pi_div_matches:
            expressions.append(f"{num}π/{denom}")

        # Extract expressions like π/2
        pi_simple_div_matches = re.findall(r"π/(\d+)", text_with_pi)
        for denom in pi_simple_div_matches:
            expressions.append(f"π/{denom}")

        # Extract expressions like 2π
        pi_mult_matches = re.findall(r"(\d+)π", text_with_pi)
        for num in pi_mult_matches:
            expressions.append(f"{num}π")

        # Extract standalone π
        if "π" in text_with_pi:
            expressions.append("π")

    def _extract_numbers(self, text: str, expressions: List[str]) -> None:
        """Extract all numbers"""
        numbers = re.findall(r"-?\d*\.?\d+", text)
        expressions.extend(numbers)

    # rule-based reward model does not need training, thus the following methods are empty
    def train(self):
        """Training method (not needed for rule-based reward model)"""
        pass

    def collect_data(self, data: list) -> None:
        """Data collection method (not needed for rule-based reward model)"""
        pass

    def clear_data(self) -> None:
        """Data clearing method (not needed for rule-based reward model)"""
        pass

    def _extract_python_code(self, text: str) -> Optional[str]:
        """Extract Python code blocks from text"""
        if text is None or not text.strip():
            return None
        # Match code between ```python and ```
        code_blocks = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        # Match code between ``` and ``` (without specified language)
        code_blocks = re.findall(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()

        return None


def strip_sequence(text: str, pad_token: str, eos_token: str) -> str:
    """
    Overview:
        Remove leading and trailing sequences of padding/eos tokens from a text.
    .. note::
        This function uses regular expressions to strip all consecutive occurrences
        of the specified padding and end-of-sequence tokens from both the beginning
        and end of the input text. Tokens in the middle of the text are preserved.
    Arguments:
        - text (str): The input text to be processed.
        - pad_token (str): The padding token to be stripped (e.g., "<PAD>").
        - eos_token (str): The end-of-sequence token to be stripped (e.g., "<EOS>").
    Returns:
        - cleaned_text (str): The cleaned text with leading/trailing padding/eos tokens removed.
    Examples:
        >>> strip_sequence("<PAD><EOS>Hello<EOS><PAD>", "<PAD>", "<EOS>")
        'Hello'
        >>> strip_sequence("Test<EOS>Middle<PAD>Keep", "<PAD>", "<EOS>")
        'Test<EOS>Middle<PAD>Keep'
        >>> strip_sequence("<EOS><EOS><PAD>Full removal<PAD><EOS>", "<PAD>", "<EOS>")
        'Full removal'
        >>> strip_sequence("No tokens here", "<PAD>", "<EOS>")
        'No tokens here'
        >>> strip_sequence("<PAD><PAD>", "<PAD>", "<EOS>")
        ''
    """
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    # Remove leading tokens
    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    # Remove trailing tokens
    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


def normalize_text(text: str) -> str:
    """
    Overview:
        This function is designed to standardize text by:
        - Converting all text to lowercase
        - Replacing various punctuation marks and special characters with spaces
        - Removing import statements
        - Normalizing whitespace by replacing multiple spaces with a single space
        - Stripping leading and trailing whitespace
    Arguments:
        - text (str): The input text to be processed.
    Returns:
        - normalized_text (str): The normalized text.
    """
    text = re.sub(r"import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
