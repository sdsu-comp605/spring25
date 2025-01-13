# Syllabus

## Logistics

*Course name*: Comp-605

*Course term*: Spring 2025

*Class time*: **M-W-F: 9-9:50 am**

*Mode of delivery*: **in person**

*Location*: **LH 445**


### Instructor: [Valeria Barra](https://valeriabarra.org)

Pronouns: (she/her/hers)

Email: vbarra @ sdsu . edu

Office location: announced on the course [Canvas page](https://sdsu.instructure.com/courses/171936)


**Office Hours**: announced on the course [Canvas page](https://sdsu.instructure.com/courses/171936) or by appointment.

:::{tip}
Office hours are an important time for asking questions, solving problems, discussing broader academic and career strategies, and providing feedback so I can make the class serve your needs and those of people with similar experiences and interests.
:::

## Overview

This course will develop the skills necessary to reason about performance of applications and modern architectures, to identify
opportunities and side-effects of changes, to develop high-performance software, to transfer algorithmic patterns and lessons learned from different domains, and to communicate such analyses with diverse
stakeholders.

These skills are important for research and development of numerical methods and performance-sensitive science and engineering
applications, or obtaining allocations via, e.g., NSF's [XSEDE](https://www.xsede.org/) and DOE [ASCR facilities](https://science.osti.gov/ascr/Facilities/Accessing-ASCR-Facilities), as well as in jobs affiliated with computing facilities at National Labs, (see for instance this [ANL careers page](https://www.alcf.anl.gov/about/careers)), industry, and academia.

We will introduce widely-used parallel programming models such as OpenMP, MPI, and CUDA, as well as ubiquitous parallel libraries, but the purpose of the course is not to teach interfaces, but to develop skills that will be durable and transferrable across different paradigms.


## Organization and course design

We will start by giving an introduction to version control and reproducibility, which are key aspects of modern computational sciences. These will serve as the foundation for the workflow that students will follow for all of their assignments during the course. We will then introduce the Linux filesystem and basic shell commands.

This course does not assume prior experience with parallel programming. It will use Linux command-line tools, and some activities will involve batch computing environments (SLURM). Most exercises and class demos will use the C and Julia programming languages, though you can use any appropriate language for projects. Some of the exercises will involve techniques and methods from numerical computing (e.g., COMP 526). I will do my best to avoid assuming prior knowledge of these topics, and to provide resources for you to learn or refresh your memory as we use them.

## Student Learning Outcomes

Upon completing this course, students will be able to

1. contribute to collaborative software with the use of version control systems, such as `git`
2. explore some numerical linear algebra with focus on efficiency of algorithms
3. evaluate the accuracy and performance of algorithms
4. develop effective numerical software, taking into account accuracy and cost
5. analize performance of serial codes
6. develop parallel algorithms and their implementations
7. identify various levels of parallelism in a problem
8. evaluate the applicability of using each level of parallelism
9. analize and develop distributed memory programs with the Message Passing Interface (MPI)
10. explore multithreaded programming and vectorization with OpenMP
11. define and analyze performance metrics and standards (strong/weak scaling, memory bandwidth, roofline analysis, etc)
12. use profiling tools and debuggers
13. explore GPU programming
14. write programs mostly in Julia and C

### Expectations

1. Enter with a growth mindset, practice adaptive coping, and nurture your intrinsic motivation
2. Attend class (in-person) and participate in discussions
3. Make an honest attempt at activities, projects, etc.
4. Interact with the class notebooks and read reference material
5. Individual or group projects

![](img/Henry2019-Table1.png)

## Assessment, grading policy and schedule

This class will have some homework assignments, midterm and final projects. The midterm/final project will be an individual or group project (depending on the number of students registered) and will be agreed upon with the instructor. There will be a final oral presentation for each final project. Moreover, a final report must be delivered. Instructions about what is expected for all homework assignments and final presentations as well as for the final report will be provided.

Grading breakdown:
- Participation (can include engagement in class, attendance, use of office hours, etc) (5%)
- Assignment 1 (5%): due Friday, February 7, by midnight (AOE)
- Assignment 2 (10%): due Friday, February 21, by midnight (AOE)
- Assignment 3 (10%): due Friday, March 07, by midnight (AOE)
- Assignment 4 (10%): due Friday, April 11, by midnight (AOE)
- Assignment 5 (10%): due Friday, April 25, by midnight (AOE)

- **Midterm project**: (15%): The midterm/final project choice and proposal will need to be discussed with your teacher, before its submission by Friday, March 14. Please make sure to use plenty of Office Hours to discuss your midterm/final project Proposal before submitting it. This project is broken down in two deadlines:
    * Part 1, due Friday, March 14, by midnight (AOE): Community project analysis and proposal.
    * Part 2, due Friday, March 28, by midnight (AOE): Community project contribution proposal and creation of an Issue.

- **Final project** (35%): this project is broken down in two deadlines:
    * Part 1: due Friday, April 18, by midnight (AOE): Creation of a Pull/Merge Request
    * Part 2: Monday, May 05 and Wednesday, May 07: In class oral presentations.


Assignments will be distributed no later than a week prior to the due date.

The schedule is subject to change (the instructor will announce any changes).

**Late submission and absences policy**: if you submit your assignments late, there is an increasing penalty (10% off for up to 24 hours late, 20% off for 24-48 hours late). No assignments will be graded if submitted later than 48 hours late.

Any student who cannot attend class or submit assignments by their due date for serious issues (e.g., medical emergencies) or participation in university activities (e.g., official university travel for conferences or sports) that can be documented, should communicate those to your instructor as soon as possible before the deadline.

## GitHub

We'll use Git with GitHub Classroom for assignments and feedback.

:::{tip}
If you don't have a GitHub account, follow these [instructions](https://sdsu-research-ci.github.io/github/students/creating-account) from the SDSU Research & Cyberinfrastructure [website](https://sdsu-research-ci.github.io/github) and [link it to your SDSUid](https://sdsu-research-ci.github.io/github/students/creating-account#linking-your-sdsuid).
- Use a personal email account rather than the SDSU one, so that you won't have problems accessing your GitHub account in the future.
- Choose your username wisely! Most likely you will use this again in professional settings in your career.
:::

We will use Canvas for announcements and grades posting. Please do _not_ submit your assignments in Canvas.

## Course materials, programming languages and environment

I will provide all free course materials and suggested readings on the [class website](https://sdsu-comp605.github.io/spring25/). If you prefer to read a print-out version, please talk to me. I will use a combination of programming languages for lectures and activities in class.

Most HPC facilities use a Linux operating system and many open source software packages and libraries will have the best documentation and testing on Linux systems. You can use any environment for your local development environment, or use the SDSU's [JupyterHub](https://jupyterhub.sdsu.edu/) to experiment and develop without a local install. If you have never logged in before, check SDSU's Research & Cyberinfrastructure [resources for students](https://sdsu-research-ci.github.io/instructionalcluster/students).

## Target audience, Preparation and Prerequisites

This course does not assume prior experience with parallel programming.

It will use Linux command-line tools, and some activities will involve batch computing environments (SLURM).

Prior exposure to the Linux/Unix operating system is useful, but not required (we will give a brief introduction). Familiarity with numerical linear algebra and/or numerical analysis is helpful.

Some of the exercises will involve techniques from [COMP 526](https://sdsu-comp526.github.io/fall24/index.html). However, I will do my best to avoid assuming prior knowledge of these topics, and to provide resources for you to learn or refresh your memory as we use them.

Everyone here is capable of succeeding in the course, but the effort level will be higher if most of the topics above are new to you.

Regardless of your preparation, it is normal to feel lost sometimes. A big part of pragmatic HPC is learning to efficiently answer your questions through documentation, online resources, and even consulting the code or running experiments. (Most of our software stack is open source.)  That said, it's easy to lose lots of time in a rabbit hole.

My hope is that you will have the courage to dive into that rabbit hole occasionally, but also to ask questions when stuck and to budget your time for such excursions so that you can complete assignments on-time without compromising your work/life balance.

The target audience is comprised of students in computational science, applied mathematics, or a quantitative science or engineering field.

Catalog Prerequisites:

* Graduate standing and knowledge of the C programming language or FORTRAN or [COMP 526](https://sdsu-comp526.github.io/fall24/index.html).

Good to know:

* Numerical Linear Algebra
* Domain Decomposition

## Classroom Behavior

Both students and faculty are responsible for maintaining an appropriate learning environment in all instructional settings, whether in person, remote, or online. Those who fail to adhere to such behavioral standards may be subject to discipline. Professional courtesy and sensitivity are especially important with respect to individuals and topics dealing with race, color, national origin, sex, pregnancy, age, disability (visible or invisible), creed, religion, sexual orientation, gender identity, gender expression, veteran status, political affiliation, or political philosophy.

## Resources for students
Every student is encouraged to read the [SDSU Student Academic Success Handbook](https://docs.google.com/document/d/1rXNpNGs1K7nIxcS73o6R-fxZqPIWQwS9gHD7XpIqjhM/edit#heading=h.apbuhr7p11ak) (includes essential information for students). Please, watch this [video](https://drive.google.com/file/d/1eViqiJ3TDjuA6a-342aLZMpFTzGCXzUJ/view?usp=drivesdk
).

## Accommodation for Disabilities

If you think you may qualify for accommodations because of a disability, please contact [SDSU Student Ability Success Center](https://sds.sdsu.edu/) and make your faculty member aware in a timely manner so that your needs can be addressed. Please allow 10-14 business days for this process.


## Preferred Student Names and Pronouns

We recognize that students' legal information doesn't always align with how they identify. Class rosters are provided to the instructor with the student's legal name. If you feel that the name that appears on the class roster does not reflect your preferred name or pronoun, let your faculty member know.

## Academic Honesty

SDSU has strict codes of conduct and policies regarding [cheating and plagiarism](https://sacd.sdsu.edu/student-rights/academic-dishonesty/cheating-and-plagiarism). Become familiar with the policy and what constitutes plagiarism. Any cheating or plagiarism will result in failing this class and a disciplinary review by the University. These actions may lead to probation, suspension, or expulsion.

This course requires you to complete various assignments that assess your understanding and application of the course content. You are expected to do your own work and cite any sources you use and collaborators appropriately. You are personally responsible for understanding and verifying the code that you submit and include appropriate documentation.

## Use of AI

In May 2024, the University Senate extended its definition of plagiarism to include the un-cited use of generative AI applications, specifically: "representing work produced by generative Artificial Intelligence as one’s own." Academic freedom ensures that instructors are empowered to determine whether students may use genAI in their classes and to what extent. To minimize confusion, we report here a statement regarding the use of AI in this class.

Students should not use generative AI applications in this course except as approved by the instructor and cited. Any use of generative AI outside of instructor-approved guidelines constitutes misuse. Misuse of generative AI is a violation of the course policy on academic honesty and will be reported to the Center for Student Rights and Responsibilities

## Sexual Misconduct, Discrimination, Harassment and/or Related Retaliation

SDSU is committed to fostering an inclusive and welcoming learning, working, and living environment. SDSU will not tolerate acts of sexual misconduct (harassment, exploitation, and assault), intimate partner violence (dating or domestic violence), stalking, or protected-class discrimination or harassment by or against members of our community. Individuals who believe they have been subject to misconduct or retaliatory actions for reporting a concern should contact the [SDSU Title IX Office](https://titleix.sdsu.edu/report-an-incident-landing).

Please know that faculty and responsible employees have a responsibility to inform the Title IX Office when made aware of incidents of sexual misconduct, dating and domestic violence, stalking, discrimination, harassment, and/or related retaliation, to ensure that individuals impacted receive information about their rights, support resources, and reporting options.

## Religious Holidays

According to the University Policy File, students should notify instructors of planned absences for religious observances by the end of the second week of classes. See the campus policy regarding religious observances for full details.

## Land Acknowledgment

For millennia, the Kumeyaay people have been a part of this land. This land has nourished, healed, protected and embraced them for many generations in a relationship of balance and harmony. As members of the San Diego State University community, we acknowledge this legacy. We promote this balance and harmony. We find inspiration from this land, the land of the Kumeyaay.

