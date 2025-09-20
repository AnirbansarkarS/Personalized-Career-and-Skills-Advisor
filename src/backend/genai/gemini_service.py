import google.generativeai as genai
from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiService:
    """Service class for interacting with Google's Gemini AI"""
    
    def __init__(self, api_key: Optional[str] = None):

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") # add api key ... i will get from darshan
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
       
        self.model = genai.GenerativeModel('gemini-pro')
        
        self.prompts = self._load_prompts()
        
        # Generation config
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
    
    def _load_prompts(self) -> Dict[str, str]:
        # Load prompt templates from files
        prompts = {}
        prompt_dir = Path(__file__).parent / "prompts"
        
        try:

            career_prompt_path = prompt_dir / "career_prompt.txt"
            if career_prompt_path.exists():
                with open(career_prompt_path, 'r') as f:
                    prompts['career'] = f.read()
            
            skill_prompt_path = prompt_dir / "skill_gap_prompt.txt"
            if skill_prompt_path.exists():
                with open(skill_prompt_path, 'r') as f:
                    prompts['skill_gap'] = f.read()
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            # backup default prompts
            prompts = self._get_default_prompts()
        
        return prompts
    
    def _get_default_prompts(self) -> Dict[str, str]:
        """Default prompts if files are not available"""
        return {
            'career': """You are an expert career advisor AI. Analyze the user profile and provide personalized career recommendations.
            
User Profile: {profile}

Please provide:
1. Top 5 recommended career paths based on skills and interests
2. Required skills for each path
3. Industry trends relevant to the user
4. Actionable next steps
5. Estimated timeline for career transition

Format the response as a structured JSON with clear recommendations.""",
            
            'skill_gap': """You are a skill development expert AI. Analyze the user's current skills and identify gaps for their target role.

Current Profile: {profile}
Target Role: {target_role}

Please provide:
1. Skill gap analysis comparing current vs required skills
2. Priority order for skill development
3. Specific learning resources and paths
4. Estimated time to acquire each skill
5. Practical projects to demonstrate skills

Format the response as structured JSON with actionable recommendations."""
        }
    
    async def generate_career_advice(
        self, 
        user_profile: Dict[str, Any],
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate personalized career advice using Gemini
        
        Args:
            user_profile: User's profile data
            additional_context: Any additional context for better recommendations
        
        Returns:
            Dictionary containing career recommendations
        """
        try:
            # Prepare the prompt
            prompt_template = self.prompts.get('career', self._get_default_prompts()['career'])
            
            # Add additional context if provided
            profile_str = json.dumps(user_profile, indent=2)
            if additional_context:
                profile_str += f"\n\nAdditional Context: {additional_context}"
            
            prompt = prompt_template.format(profile=profile_str)
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Parse response
            result = self._parse_response(response.text)
            
            return {
                "status": "success",
                "recommendations": result,
                "confidence_score": 0.85
            }
            
        except Exception as e:
            logger.error(f"Error generating career advice: {e}")
            return {
                "status": "error",
                "message": str(e),
                "recommendations": None
            }
    
    async def analyze_skill_gaps(
        self,
        user_profile: Dict[str, Any],
        target_role: str,
        industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze skill gaps between current profile and target role
        
        Args:
            user_profile: User's current profile
            target_role: Target job role
            industry: Industry context
        
        Returns:
            Dictionary containing skill gap analysis
        """
        try:
            # Prepare the prompt
            prompt_template = self.prompts.get('skill_gap', self._get_default_prompts()['skill_gap'])
            
            profile_str = json.dumps(user_profile, indent=2)
            if industry:
                profile_str += f"\nIndustry: {industry}"
            
            prompt = prompt_template.format(
                profile=profile_str,
                target_role=target_role
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Parse response
            result = self._parse_response(response.text)
            
            return {
                "status": "success",
                "analysis": result,
                "target_role": target_role,
                "confidence_score": 0.88
            }
            
        except Exception as e:
            logger.error(f"Error analyzing skill gaps: {e}")
            return {
                "status": "error",
                "message": str(e),
                "analysis": None
            }
    
    async def generate_learning_path(
        self,
        skills_to_learn: List[str],
        current_level: str,
        time_commitment: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Generate a personalized learning path for skill development
        
        Args:
            skills_to_learn: List of skills to acquire
            current_level: Current skill level (beginner/intermediate/advanced)
            time_commitment: Time available (low/moderate/high)
        
        Returns:
            Dictionary containing learning path recommendations
        """
        try:
            prompt = f"""Create a detailed learning path for the following skills:

Skills to Learn: {', '.join(skills_to_learn)}
Current Level: {current_level}
Time Commitment: {time_commitment}

Please provide:
1. Structured learning roadmap with milestones
2. Recommended resources (courses, books, projects)
3. Time estimates for each milestone
4. Practice projects to reinforce learning
5. Assessment criteria for skill validation

Format as structured JSON with clear timeline and resources."""

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            result = self._parse_response(response.text)
            
            return {
                "status": "success",
                "learning_path": result,
                "estimated_duration": self._estimate_duration(skills_to_learn, current_level)
            }
            
        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            return {
                "status": "error",
                "message": str(e),
                "learning_path": None
            }
    
    async def get_industry_insights(
        self,
        industry: str,
        role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current industry insights and trends
        
        Args:
            industry: Industry name
            role: Specific role within industry
        
        Returns:
            Dictionary containing industry insights
        """
        try:
            role_context = f" for {role}" if role else ""
            
            prompt = f"""Provide current industry insights and trends for {industry}{role_context}.

Include:
1. Top 5 current trends in the industry
2. Emerging skills in demand
3. Future outlook (next 2-5 years)
4. Key companies and opportunities
5. Salary ranges and growth potential

Format as structured JSON with actionable insights."""

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            result = self._parse_response(response.text)
            
            return {
                "status": "success",
                "insights": result,
                "industry": industry,
                "role": role
            }
            
        except Exception as e:
            logger.error(f"Error getting industry insights: {e}")
            return {
                "status": "error",
                "message": str(e),
                "insights": None
            }
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse AI response and extract structured data
        
        Args:
            response_text: Raw response from AI model
        
        Returns:
            Parsed dictionary or formatted response
        """
        try:
            # Try to parse as JSON first
            # Remove any markdown formatting if present
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            # If not JSON, structure the response
            return {
                "content": response_text,
                "format": "text"
            }
    
    def _estimate_duration(self, skills: List[str], current_level: str) -> str:
        """
        Estimate learning duration based on skills and current level
        
        Args:
            skills: List of skills to learn
            current_level: Current skill level
        
        Returns:
            Estimated duration string
        """
        base_hours = {
            "beginner": 100,
            "intermediate": 60,
            "advanced": 30
        }
        
        hours_per_skill = base_hours.get(current_level, 80)
        total_hours = len(skills) * hours_per_skill
        
        if total_hours < 100:
            return f"{total_hours} hours (1-2 months)"
        elif total_hours < 300:
            return f"{total_hours} hours (2-4 months)"
        elif total_hours < 600:
            return f"{total_hours} hours (4-6 months)"
        else:
            return f"{total_hours} hours (6-12 months)"