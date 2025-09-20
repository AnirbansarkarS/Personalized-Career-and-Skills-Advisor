from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SkillsMapper:
    """Advanced skills mapping and recommendation engine"""
    
    def __init__(self):
        """Initialize the skills mapper with predefined skill relationships"""
        self.skill_taxonomy = self._build_skill_taxonomy()
        self.career_skill_matrix = self._build_career_skill_matrix()
        self.skill_relationships = self._build_skill_relationships()
        self.learning_prerequisites = self._build_prerequisites()
    
    def _build_skill_taxonomy(self) -> Dict[str, List[str]]:
        """Build hierarchical skill taxonomy"""
        return {
            "technical": {
                "programming": [
                    "Python", "JavaScript", "Java", "C++", "Go", "Rust",
                    "TypeScript", "SQL", "R", "Swift", "Kotlin"
                ],
                "data_science": [
                    "Machine Learning", "Deep Learning", "Statistics",
                    "Data Analysis", "Data Visualization", "Big Data",
                    "NLP", "Computer Vision", "Time Series Analysis"
                ],
                "cloud": [
                    "AWS", "Azure", "GCP", "Docker", "Kubernetes",
                    "Terraform", "CI/CD", "Microservices"
                ],
                "ai_ml": [
                    "TensorFlow", "PyTorch", "Scikit-learn", "Keras",
                    "Hugging Face", "LangChain", "OpenAI API", "MLOps"
                ],
                "web_dev": [
                    "React", "Angular", "Vue.js", "Node.js", "Django",
                    "Flask", "FastAPI", "HTML/CSS", "REST APIs", "GraphQL"
                ],
                "database": [
                    "PostgreSQL", "MongoDB", "MySQL", "Redis",
                    "Elasticsearch", "Cassandra", "Neo4j"
                ]
            },
            "soft_skills": {
                "leadership": [
                    "Team Management", "Strategic Thinking", "Decision Making",
                    "Conflict Resolution", "Delegation", "Mentoring"
                ],
                "communication": [
                    "Presentation Skills", "Technical Writing", "Stakeholder Management",
                    "Active Listening", "Negotiation", "Public Speaking"
                ],
                "analytical": [
                    "Problem Solving", "Critical Thinking", "Research",
                    "Data Interpretation", "System Thinking", "Risk Assessment"
                ],
                "project": [
                    "Project Management", "Agile", "Scrum", "Kanban",
                    "Risk Management", "Resource Planning"
                ]
            },
            "domain": {
                "finance": [
                    "Financial Analysis", "Risk Management", "Trading",
                    "Blockchain", "Cryptocurrency", "Quantitative Finance"
                ],
                "healthcare": [
                    "HIPAA Compliance", "Clinical Data", "Medical Imaging",
                    "Bioinformatics", "Healthcare Analytics", "EHR Systems"
                ],
                "marketing": [
                    "Digital Marketing", "SEO/SEM", "Content Marketing",
                    "Marketing Analytics", "Social Media", "Brand Management"
                ]
            }
        }
    
    def _build_career_skill_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build matrix mapping careers to required skills with importance scores"""
        return {
            "Data Scientist": {
                "Python": 1.0, "Machine Learning": 1.0, "Statistics": 0.9,
                "SQL": 0.8, "Data Visualization": 0.8, "Deep Learning": 0.7,
                "Communication": 0.7, "Problem Solving": 0.9
            },
            "ML Engineer": {
                "Python": 1.0, "Machine Learning": 1.0, "Deep Learning": 0.9,
                "MLOps": 0.9, "Docker": 0.8, "Kubernetes": 0.7,
                "Cloud": 0.8, "Software Engineering": 0.9
            },
            "Full Stack Developer": {
                "JavaScript": 1.0, "React": 0.9, "Node.js": 0.9,
                "SQL": 0.8, "REST APIs": 0.9, "Git": 0.9,
                "Problem Solving": 0.9, "HTML/CSS": 1.0
            },
            "DevOps Engineer": {
                "Docker": 1.0, "Kubernetes": 1.0, "CI/CD": 1.0,
                "AWS": 0.9, "Terraform": 0.8, "Python": 0.7,
                "Monitoring": 0.9, "Linux": 0.9
            },
            "Product Manager": {
                "Product Strategy": 1.0, "Stakeholder Management": 0.9,
                "Data Analysis": 0.8, "Communication": 1.0,
                "Agile": 0.8, "User Research": 0.9, "Leadership": 0.8
            },
            "AI Engineer": {
                "Python": 1.0, "Deep Learning": 1.0, "PyTorch": 0.9,
                "TensorFlow": 0.9, "NLP": 0.8, "Computer Vision": 0.8,
                "MLOps": 0.7, "Cloud": 0.8, "LangChain": 0.7
            },
            "Data Engineer": {
                "Python": 0.9, "SQL": 1.0, "Spark": 0.9,
                "ETL": 1.0, "Data Warehousing": 0.9, "Cloud": 0.8,
                "Airflow": 0.8, "Big Data": 0.9
            },
            "Cloud Architect": {
                "AWS": 1.0, "Azure": 0.8, "GCP": 0.8,
                "Architecture Design": 1.0, "Security": 0.9,
                "Networking": 0.8, "Cost Optimization": 0.8,
                "IaC": 0.9
            }
        }
    
    def _build_skill_relationships(self) -> Dict[str, List[str]]:
        """Build relationships between skills (complementary skills)"""
        return {
            "Python": ["Data Analysis", "Machine Learning", "Django", "Flask", "FastAPI"],
            "Machine Learning": ["Deep Learning", "Statistics", "Python", "Data Analysis"],
            "React": ["JavaScript", "TypeScript", "Redux", "Node.js", "HTML/CSS"],
            "AWS": ["Cloud Architecture", "DevOps", "Terraform", "Docker"],
            "Docker": ["Kubernetes", "CI/CD", "Microservices", "DevOps"],
            "Data Analysis": ["SQL", "Python", "Statistics", "Data Visualization"],
            "JavaScript": ["TypeScript", "React", "Node.js", "Vue.js", "Angular"],
            "SQL": ["Database Design", "Data Analysis", "PostgreSQL", "MySQL"]
        }
    
    def _build_prerequisites(self) -> Dict[str, List[str]]:
        """Build prerequisite skills for advanced skills"""
        return {
            "Machine Learning": ["Python", "Statistics", "Linear Algebra"],
            "Deep Learning": ["Machine Learning", "Python", "Calculus"],
            "React": ["JavaScript", "HTML/CSS"],
            "Kubernetes": ["Docker", "Linux", "Networking"],
            "MLOps": ["Machine Learning", "Docker", "CI/CD"],
            "Data Engineering": ["SQL", "Python", "Database Design"],
            "Cloud Architecture": ["Networking", "Security", "Linux"],
            "NLP": ["Machine Learning", "Python", "Statistics"]
        }
    
    def calculate_skill_similarity(self, skill1: str, skill2: str) -> float:
        """
        Calculate similarity score between two skills
        
        Args:
            skill1: First skill
            skill2: Second skill
        
        Returns:
            Similarity score between 0 and 1
        """
        if skill1 == skill2:
            return 1.0
        
        # Check direct relationships
        relationships = self.skill_relationships.get(skill1, [])
        if skill2 in relationships:
            return 0.8
        
        # Check category similarity
        skill1_category = self._find_skill_category(skill1)
        skill2_category = self._find_skill_category(skill2)
        
        if skill1_category == skill2_category and skill1_category:
            return 0.6
        
        # Check for common prerequisites
        prereq1 = set(self.learning_prerequisites.get(skill1, []))
        prereq2 = set(self.learning_prerequisites.get(skill2, []))
        
        if prereq1 and prereq2:
            common = len(prereq1.intersection(prereq2))
            total = len(prereq1.union(prereq2))
            if total > 0:
                return common / total * 0.5
        
        return 0.0
    
    def _find_skill_category(self, skill: str) -> Optional[str]:
        """Find the category of a skill in the taxonomy"""
        for main_category, subcategories in self.skill_taxonomy.items():
            for subcategory, skills in subcategories.items():
                if skill in skills:
                    return f"{main_category}.{subcategory}"
        return None
    
    def map_skills_to_careers(
        self,
        user_skills: Dict[str, str]
    ) -> List[Dict[str, any]]:
        """
        Map user skills to potential career paths
        
        Args:
            user_skills: Dictionary of user skills with proficiency levels
        
        Returns:
            List of career matches with scores
        """
        career_scores = []
        
        for career, required_skills in self.career_skill_matrix.items():
            score = 0
            matched_skills = []
            missing_skills = []
            
            for req_skill, importance in required_skills.items():
                if req_skill in user_skills:
                    # Direct match
                    proficiency_multiplier = self._get_proficiency_multiplier(
                        user_skills[req_skill]
                    )
                    score += importance * proficiency_multiplier
                    matched_skills.append(req_skill)
                else:
                    # Check for similar skills
                    similar_skill = self._find_similar_skill(
                        req_skill, list(user_skills.keys())
                    )
                    if similar_skill:
                        similarity = self.calculate_skill_similarity(
                            req_skill, similar_skill
                        )
                        proficiency_multiplier = self._get_proficiency_multiplier(
                            user_skills[similar_skill]
                        )
                        score += importance * similarity * proficiency_multiplier * 0.7
                        matched_skills.append(f"{similar_skill} (similar to {req_skill})")
                    else:
                        missing_skills.append(req_skill)
            
            # Calculate match percentage
            match_percentage = (score / sum(required_skills.values())) * 100
            
            career_scores.append({
                "career": career,
                "match_score": round(match_percentage, 2),
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "skill_gap_count": len(missing_skills),
                "recommendation_level": self._get_recommendation_level(match_percentage)
            })
        
        # Sort by match score
        career_scores.sort(key=lambda x: x["match_score"], reverse=True)
        
        return career_scores[:10]  # Return top 10 matches
    
    def _get_proficiency_multiplier(self, level: str) -> float:
        """Convert proficiency level to numerical multiplier"""
        multipliers = {
            "beginner": 0.25,
            "intermediate": 0.5,
            "advanced": 0.75,
            "expert": 1.0
        }
        return multipliers.get(level.lower(), 0.5)
    
    def _get_recommendation_level(self, match_score: float) -> str:
        """Determine recommendation level based on match score"""
        if match_score >= 80:
            return "Excellent Match"
        elif match_score >= 60:
            return "Good Match"
        elif match_score >= 40:
            return "Potential Match"
        else:
            return "Stretch Opportunity"
    
    def _find_similar_skill(self, target_skill: str, user_skills: List[str]) -> Optional[str]:
        """Find the most similar skill from user's skills"""
        max_similarity = 0
        most_similar = None
        
        for skill in user_skills:
            similarity = self.calculate_skill_similarity(target_skill, skill)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar = skill
        
        return most_similar if max_similarity > 0.4 else None
    
    def identify_skill_gaps(
        self,
        current_skills: Dict[str, str],
        target_role: str
    ) -> Dict[str, any]:
        """
        Identify skill gaps for a target role
        
        Args:
            current_skills: User's current skills with levels
            target_role: Target career role
        
        Returns:
            Detailed skill gap analysis
        """
        if target_role not in self.career_skill_matrix:
            # Find closest match
            target_role = self._find_closest_role(target_role)
        
        required_skills = self.career_skill_matrix.get(target_role, {})
        
        gaps = {
            "critical": [],  # High importance
            "important": [],  # Medium 
            "nice_to_have": [],  # Low imp
            "needs_improvement": []  #needs higher proficiency
        }
        
        for skill, importance in required_skills.items():
            if skill in current_skills:
                current_level = self._get_proficiency_multiplier(current_skills[skill])
                if current_level < importance * 0.7:  # Needs improvement
                    gaps["needs_improvement"].append({
                        "skill": skill,
                        "current_level": current_skills[skill],
                        "required_level": self._get_level_from_importance(importance),
                        "importance": importance
                    })
            else:
                # Skill is missing
                if importance >= 0.9:
                    gaps["critical"].append({
                        "skill": skill,
                        "importance": importance,
                        "prerequisites": self.learning_prerequisites.get(skill, [])
                    })
                elif importance >= 0.7:
                    gaps["important"].append({
                        "skill": skill,
                        "importance": importance,
                        "prerequisites": self.learning_prerequisites.get(skill, [])
                    })
                else:
                    gaps["nice_to_have"].append({
                        "skill": skill,
                        "importance": importance,
                        "prerequisites": self.learning_prerequisites.get(skill, [])
                    })
        
        return {
            "target_role": target_role,
            "skill_gaps": gaps,
            "total_gaps": sum(len(v) for v in gaps.values()),
            "readiness_score": self._calculate_readiness(current_skills, required_skills),
            "estimated_time_to_ready": self._estimate_learning_time(gaps)
        }
    
    def _find_closest_role(self, role_name: str) -> str:
        """Find the closest matching role in the career matrix"""
        role_lower = role_name.lower()
        for career in self.career_skill_matrix.keys():
            if role_lower in career.lower() or career.lower() in role_lower:
                return career
        return list(self.career_skill_matrix.keys())[0]  # Default to first
    
    def _get_level_from_importance(self, importance: float) -> str:
        """Convert importance score to proficiency level"""
        if importance >= 0.9:
            return "expert"
        elif importance >= 0.7:
            return "advanced"
        elif importance >= 0.5:
            return "intermediate"
        else:
            return "beginner"
    
    def _calculate_readiness(
        self,
        current_skills: Dict[str, str],
        required_skills: Dict[str, float]
    ) -> float:
        """Calculate readiness score for a role"""
        if not required_skills:
            return 0.0
        
        total_score = 0
        total_weight = sum(required_skills.values())
        
        for skill, importance in required_skills.items():
            if skill in current_skills:
                level_multiplier = self._get_proficiency_multiplier(current_skills[skill])
                total_score += importance * level_multiplier
        
        return round((total_score / total_weight) * 100, 2) if total_weight > 0 else 0.0
    
    def _estimate_learning_time(self, gaps: Dict[str, List]) -> str:
        """Estimate time needed to fill skill gaps"""
        # Approximate hours for each skill level
        hours_per_skill = {
            "critical": 200,
            "important": 150,
            "nice_to_have": 100,
            "needs_improvement": 50
        }
        
        total_hours = 0
        for gap_type, skills in gaps.items():
            total_hours += len(skills) * hours_per_skill.get(gap_type, 100)
        
        # Convert to months (assuming 20 hours/week dedication)
        months = total_hours / 80  
        
        if months < 3:
            return f"{round(months, 1)} months"
        elif months < 12:
            return f"{round(months)} months"
        else:
            years = months / 12
            return f"{round(years, 1)} years"
    
    def get_learning_sequence(
        self,
        skills_to_learn: List[str]
    ) -> List[Dict[str, any]]:
        """
        Generate optimal learning sequence considering prerequisites
        
        Args:
            skills_to_learn: List of target skills
        
        Returns:
            Ordered learning sequence with phases
        """
        # Build dependency graph
        dependencies = {}
        all_skills = set(skills_to_learn)
        
        # Add all prerequisites
        for skill in skills_to_learn:
            prereqs = self.learning_prerequisites.get(skill, [])
            dependencies[skill] = prereqs
            all_skills.update(prereqs)
        
        # Topological sort for learning order
        learning_order = self._topological_sort(dependencies, list(all_skills))
        
        # Group into learning phases
        phases = []
        phase_num = 1
        processed = set()
        
        for skills_batch in self._group_into_phases(learning_order, dependencies):
            phase_skills = []
            for skill in skills_batch:
                if skill in all_skills and skill not in processed:
                    phase_skills.append({
                        "skill": skill,
                        "is_target": skill in skills_to_learn,
                        "estimated_hours": self._estimate_skill_hours(skill),
                        "resources": self._get_learning_resources(skill)
                    })
                    processed.add(skill)
            
            if phase_skills:
                phases.append({
                    "phase": phase_num,
                    "skills": phase_skills,
                    "duration": f"{sum(s['estimated_hours'] for s in phase_skills)} hours",
                    "focus": "Prerequisites" if phase_num == 1 else "Core Skills"
                })
                phase_num += 1
        
        return phases
    
    def _topological_sort(
        self,
        dependencies: Dict[str, List[str]],
        all_skills: List[str]
    ) -> List[str]:
        """Perform topological sort on skill dependencies"""
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for skill in all_skills:
            in_degree[skill] = 0
        
        for skill, prereqs in dependencies.items():
            for prereq in prereqs:
                graph[prereq].append(skill)
                in_degree[skill] += 1
        
        # Find all nodes with no prerequisites
        queue = [skill for skill in all_skills if in_degree[skill] == 0]
        result = []
        
        while queue:
            skill = queue.pop(0)
            result.append(skill)
            
            # Remove this skill from dependencies
            for dependent in graph[skill]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return result
    
    def _group_into_phases(
        self,
        learning_order: List[str],
        dependencies: Dict[str, List[str]]
    ) -> List[List[str]]:
        """Group skills into learning phases"""
        phases = []
        processed = set()
        
        while len(processed) < len(learning_order):
            current_phase = []
            for skill in learning_order:
                if skill not in processed:
                    # Check if all prerequisites are processed
                    prereqs = dependencies.get(skill, [])
                    if all(p in processed for p in prereqs):
                        current_phase.append(skill)
            
            if current_phase:
                phases.append(current_phase)
                processed.update(current_phase)
            else:
                break  # Avoid infinite loop
        
        return phases
    
    def _estimate_skill_hours(self, skill: str) -> int:
        """Estimate learning hours for a skill"""
        # Base estimates by category
        category = self._find_skill_category(skill)
        
        if category:
            if "programming" in category:
                return 150
            elif "data_science" in category or "ai_ml" in category:
                return 200
            elif "cloud" in category:
                return 120
            elif "soft_skills" in category:
                return 80
            elif "web_dev" in category:
                return 100
        
        return 100  # Default
    
    def _get_learning_resources(self, skill: str) -> List[str]:
        """Get recommended learning resources for a skill"""
        
        # For now, returning generic resources (if database added then from there...)
        resources = {
            "Python": ["Python Crash Course", "Automate the Boring Stuff", "Python.org Tutorial"],
            "Machine Learning": ["Andrew Ng's Course", "Fast.ai", "Hands-On ML Book"],
            "React": ["React Official Docs", "React Tutorial", "Full Stack Open"],
            "AWS": ["AWS Training", "Cloud Practitioner Cert", "AWS Docs"],
            "Docker": ["Docker Docs", "Docker Mastery Course", "Play with Docker"],
        }
        
        return resources.get(skill, ["Online Tutorials", "Documentation", "Practice Projects"])
    
    def recommend_complementary_skills(
        self,
        current_skills: List[str],
        max_recommendations: int = 5
    ) -> List[Dict[str, any]]:
        """
        Recommend complementary skills based on current skillset
        
        Args:
            current_skills: List of current skills
            max_recommendations: Maximum number of recommendations
        
        Returns:
            List of recommended skills with reasoning
        """
        recommendations = {}
        
        # Find related skills
        for skill in current_skills:
            related = self.skill_relationships.get(skill, [])
            for related_skill in related:
                if related_skill not in current_skills:
                    if related_skill not in recommendations:
                        recommendations[related_skill] = {
                            "skill": related_skill,
                            "reasons": [],
                            "synergy_score": 0
                        }
                    recommendations[related_skill]["reasons"].append(
                        f"Complements {skill}"
                    )
                    recommendations[related_skill]["synergy_score"] += 1
        
        # Find skills that unlock new opportunities
        for career, required_skills in self.career_skill_matrix.items():
            skill_match = sum(1 for s in required_skills if s in current_skills)
            match_percentage = skill_match / len(required_skills) if required_skills else 0
            
            # If user has 40-70% match, recommend missing high-important   sskills
            if 0.4 <= match_percentage <= 0.7:
                for skill, importance in required_skills.items():
                    if skill not in current_skills and importance >= 0.8:
                        if skill not in recommendations:
                            recommendations[skill] = {
                                "skill": skill,
                                "reasons": [],
                                "synergy_score": 0
                            }
                        recommendations[skill]["reasons"].append(
                            f"Key skill for {career}"
                        )
                        recommendations[skill]["synergy_score"] += importance
        
        # Sort by synergy score and return top recommendations
        sorted_recommendations = sorted(
            recommendations.values(),
            key=lambda x: x["synergy_score"],
            reverse=True
        )
        
        return sorted_recommendations[:max_recommendations]
    
    def get_skill_market_demand(self, skill: str) -> Dict[str, any]:
        """
        Get market demand analysis for a skill
        
        Args:
            skill: Skill name
        
        Returns:
            Market demand information
        """
        # This would typically fetch real market data
        # For now, using predefined demand levels
        high_demand_skills = [
            "Python", "Machine Learning", "Cloud", "AWS", "React",
            "Docker", "Kubernetes", "Data Science", "AI", "DevOps"
        ]
        
        medium_demand_skills = [
            "Java", "SQL", "JavaScript", "Node.js", "Angular",
            "Project Management", "Agile", "Data Analysis"
        ]
        
        demand_level = "High" if skill in high_demand_skills else \
                      "Medium" if skill in medium_demand_skills else "Moderate"
        
        return {
            "skill": skill,
            "demand_level": demand_level,
            "trending": skill in ["AI", "Machine Learning", "Cloud", "DevOps"],
            "average_salary_impact": "+15-25%" if demand_level == "High" else "+5-15%",
            "job_postings_growth": "+30%" if demand_level == "High" else "+10%",
            "recommendation": self._get_demand_recommendation(demand_level)
        }
    
    def _get_demand_recommendation(self, demand_level: str) -> str:
        """Get recommendation based on demand level"""
        recommendations = {
            "High": "Excellent skill to learn - high market demand and career opportunities",
            "Medium": "Good skill to have - steady demand across industries",
            "Moderate": "Useful skill - consider combining with high-demand skills"
        }
        return recommendations.get(demand_level, "Consider market trends")