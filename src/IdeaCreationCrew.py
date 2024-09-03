from textwrap import dedent
from crewai import Agent, Task, Crew
from tools import ExaSearchToolset
# from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

#Call Gemini model
llm1 = ChatGoogleGenerativeAI(
    model='gemini-1.5-pro',
    verbose=True,
    temperature=0.5,
    google_api_key=os.getenv('GOOGLE_API_KEY')
)

# llm2 = ChatGroq(
#     api_key=os.getenv('GROQ_API_KEY'),
#     model="mixtral-8x7b-32768"
# )


class IdeaCreationAgents:
    def idea_creation_agent(self):
        return Agent(
            role="Creative Genius",
            goal="Come up with a new business idea that no one has thought of before!",
            tools=ExaSearchToolset.tools(),
            backstory=dedent("""\
                You are a creative genius with a passion for innovation. Your expertise lies in
                coming up with new and exciting ideas that have the potential to change the world.
                You are skilled in brainstorming, problem-solving, and thinking outside the box.
                """),
            verbose=True,
            llm=llm1,
            #max_iter=3
        )
    
    def idea_refinement_agent(self):
        return Agent(
            role="Idea Refiner",
            goal="Refine the business idea to make it more viable and successful",
            tools=ExaSearchToolset.tools(),
            backstory=dedent("""\
                You are an expert in refining business ideas to enhance their viability and success potential.
                Your skills include identifying challenges, suggesting improvements, and providing detailed
                feedback to improve the initial concept.
                """),
            verbose=True,
            llm=llm1,
        )
    
    def idea_judge_agent(self):
        return Agent(
            role="Shark Tank Judge",
            goal="Judge the viability of the idea and provide feedback",
            tools=ExaSearchToolset.tools(),
            backstory=dedent("""\
                You are a Shark Tank judge with a keen eye for spotting winning ideas. Your expertise
                lies in evaluating the potential of new business concepts and providing feedback to help
                entrepreneurs succeed. You are skilled in market analysis, financial evaluation, and strategic"
                planning. You are passionate about supporting innovation and helping businesses thrive.
                You are a tough critic, but your feedback is always constructive and valuable."""),
            verbose=True,
            llm=llm1,
            #max_iter= 3
        )
    
    def idea_pitch_agent(self):
        return Agent(
            role="Master Idea Pitcher",
            goal="Pitch the idea to potential investors or partners no matter what!",
            tools=ExaSearchToolset.tools(),
            backstory=dedent("""\
                You are a master idea pitcher with a gift for captivating audiences and winning over investors.
                Your expertise lies in crafting compelling pitches that highlight the value and potential of new
                business ideas. You are skilled in storytelling, persuasion, and presentation. You are passionate
                about sharing innovative concepts and inspiring others to join you on the journey to success.
                """),
            verbose=True,
            llm=llm1,
            #max_iter= 3
        )

class IdeaCreationTasks:
    def idea_creation_task(self, agent):
        return Task(
            description=dedent(f"""\
                Come up with a new business idea that no one has thought of before.
                The idea must be plausible and have the potential to be successful.
                The idea must be unique and innovative. The idea MUST be an implementation
                of AI technology in any industry. If the idea has been rejected 
                by the judge and it cannot be improved any more, then you will need 
                come up with a brand new idea."""),
            expected_output=dedent("""\
                A detailed description of the new business idea. The description should include
                the problem the idea solves, the target market, the value proposition, and a name
                for the business.
                                   
                Structure:
                                   
                Business Name:---
                Problem:-----
                Solution:-----
                Detailed Description:-----
                Target Market:----
                Value Proposition:-----
                """),
            agent=agent,
        )
    
    def idea_refining_task(self, agent, idea):
        return Task(
            description=dedent(f"""\
                Idea: {idea}       

                Based on the idea provided by the user, refine the idea to make it more
                viable and successful. This could involve adding more details, identifying
                potential challenges, or suggesting improvements. If the idea has been rejected 
                by the judge and it cannot be improved any more, then you will need to prompt 
                the idea creator to come up with a brand new idea."""),
            expected_output=dedent("""\
                A refined version of the business idea. The description should include any
                changes or improvements made to the original idea. This could include addressing
                potential challenges, identifying new opportunities, or refining the value proposition.
                Overall, the output should be a more detailed and well-thought-out version of the 
                original idea."""),
            agent=agent,
        )
    
    def idea_judging_task(self, agent):
        return Task(
            description=dedent("""\
                based on the idea provided by the user, determine whether the idea is viable
                and has the potential to be successful. This should involve evaluating the market
                opportunity, the competitive landscape, the value proposition, and the feasibility in 
                both monetary and technical terms. The Idea needs to be REALLY REALLY GOOD to be accepted.
                """),
            expected_output=dedent("""\
                if the idea is viable and has the REAL potential to be successful,
                then provide a 'valid' response. If the idea is not viable or has no potential to be successful,
                then provide a 'rejected' response. 
                """),
            agent=agent, 
        )

    def next_steps_task(self, agent, idea):
        return Task(
            description=dedent(f"""\
                Using the approved idea, write a detailed report on how it could be implemented.
                
                Idea: {idea}"""),
            expected_output=dedent("""\
                A detailed report on the implementation of the business idea. The report should be structured as follows:

            1. **Executive Summary:**
               - Overview of the business idea.
               - Key highlights and value proposition.

            2. **Problem Statement:**
               - Detailed description of the problem the idea aims to solve.
               - Evidence and examples illustrating the problem.

            3. **Solution:**
               - In-depth explanation of the proposed solution.
               - Unique features and benefits of the solution.

            4. **Market Analysis:**
               - Target market and customer segments.
               - Market size and growth potential.
               - Competitive landscape and differentiation.

            5. **Business Model:**
               - Revenue model and pricing strategy.
               - Key partnerships and resources.
               - Customer acquisition and retention strategies.

            6. **Implementation Plan:**
               - Step-by-step plan for bringing the idea to life.
               - Key milestones and timelines.
               - Resource requirements (e.g., team, technology, funding).

            7. **Technical Feasibility:**
               - Technical requirements and specifications.
               - Development and deployment plan.
               - Potential technical challenges and solutions.

            8. **Financial Projections:**
               - Detailed financial projections (e.g., revenue, costs, profits).
               - Break-even analysis.
               - Funding requirements and sources.

            9. **Risk Analysis:**
               - Potential risks and challenges.
               - Mitigation strategies and contingency plans.

            10. **Conclusion:**
                - Summary of the key points.
                - Final thoughts and recommendations.

            The report should be well-organized and professionally written, providing a clear roadmap for implementing the business idea.
            """),
            agent=agent,
        )
    
    
    
def iterative_idea_generation(prompt, quality_threshold=7, max_iterations=10):
    agents = IdeaCreationAgents()
    tasks = IdeaCreationTasks()

    # Initialize agents
    creator_agent = agents.idea_creation_agent()
    refiner_agent = agents.idea_refinement_agent()
    judge_agent = agents.idea_judge_agent()
    writer_agent = agents.idea_pitch_agent()

    iterations = 0
    successful_idea = None
    implementation_report = None

    while iterations < max_iterations:
        print(f"Iteration {iterations + 1}/{max_iterations}")

        # Generate new business ideas
        print("Generating new business ideas...")
        creation_task = tasks.idea_creation_task(creator_agent)
        crew = Crew(agents=[creator_agent], tasks=[creation_task])
        result = crew.kickoff()

        if not isinstance(result, dict):
            print("Unexpected result:", result)
            iterations += 1
            continue

        idea_details = {
            "Business Name": result.get("Business Name", ""),
            "Problem": result.get("Problem", ""),
            "Solution": result.get("Solution", ""),
            "Detailed Description": result.get("Detailed Description", ""),
            "Target Market": result.get("Target Market", ""),
            "Value Proposition": result.get("Value Proposition", "")
        }

        print("Evaluating idea:", idea_details)

        # Refine the idea
        print("Refining the idea...")
        refinement_task = tasks.idea_refinement_task(refiner_agent, idea_details)
        crew = Crew(agents=[refiner_agent], tasks=[refinement_task])
        refined_result = crew.kickoff()
        refined_idea = refined_result.get('output', idea_details)
        print(f"Refined idea: {refined_idea}")

        # Judge the idea
        print("Judging the idea...")
        judging_task = tasks.idea_judging_task(judge_agent)
        crew = Crew(agents=[judge_agent], tasks=[judging_task])
        judging_result = crew.kickoff()

        if not judging_result:
            print("Failed to judge idea. Exiting loop.")
            break

        judge_decision = judging_result.get("output")
        print(f"Judge decision: {judge_decision}")

        if judge_decision == 'valid':
            print("Idea approved. Writing the implementation report...")
            next_steps_task = tasks.next_steps_task(writer_agent, refined_idea)
            crew = Crew(agents=[writer_agent], tasks=[next_steps_task])
            report_result = crew.kickoff()

            if not report_result:
                print("Failed to write the report. Exiting loop.")
                break

            implementation_report = report_result.get("output")
            successful_idea = refined_idea
            print(f"Implementation report:\n{implementation_report}")
            break
        else:
            print("Idea rejected. Generating a new idea.")
        
        iterations += 1

    if successful_idea:
        print("Successful idea found and report generated.")
        return {
            "idea": successful_idea,
            "report": implementation_report
        }
    else:
        print("No viable idea found within the iteration limit.")
        return {
            "error": "No viable idea found"
        }


# Run the workflow
def run():
    prompt = "Generate business ideas related to the implementation of AI"
    final_idea, report = iterative_idea_generation(prompt)
    if final_idea:
        print("Final Idea:", final_idea)
        print("Implementation Report:", report)
    else:
        print("Failed to generate a viable idea within the iteration limit.")

if __name__ == '__main__':
    run()
