# EduPro_Student_Recommendation
A machine learning project for student segmentation and personalized course recommendations.
# EduPro – Student Segmentation and Personalized Course Recommendation System

## Project Overview

EduPro is a machine learning project that analyzes student learning behaviour, identifies learner segments, and generates personalized course recommendations.

The project combines data analytics, K-Means clustering, and a hybrid recommendation system to provide course suggestions based on student preferences and historical learning patterns.

## Objectives

- Analyze student course enrolment behaviour.
- Identify learner segments using K-Means clustering.
- Understand course category and level preferences.
- Generate personalized course recommendations.
- Evaluate recommendation performance using historical transactions.
- Develop an interactive Streamlit application.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- K-Means Clustering
- Streamlit
- Excel

## Dataset

The project dataset contains:

| Dataset | Records |
|---|---:|
| Students | 3,000 |
| Teachers | 60 |
| Courses | 60 |
| Transactions | 10,000 |

The transaction data covers January–December 2025.

## Methodology

1. Data understanding and validation
2. Exploratory Data Analysis
3. Student profile creation
4. Feature engineering and preprocessing
5. K-Means clustering
6. Hybrid course recommendation
7. Recommendation evaluation
8. Error analysis and weight optimization
9. Streamlit application development

## Student Segmentation

Four learner segments were identified:

- Advanced-focused learners
- Beginner-focused learners
- Multi-course explorers
- Intermediate-focused learners

These segment names are interpretations of the cluster profiles.

## Recommendation Approach

The hybrid recommendation score combines:

- Cluster Popularity: 50%
- Category Match: 20%
- Level Match: 20%
- Course Rating: 10%

Courses already taken by a student are excluded.

## Model Evaluation

| Method | Precision@5 | Recall@5 | F1@5 | Catalog Coverage |
|---|---:|---:|---:|---:|
| Global Popularity | 0.0417 | 0.0988 | 0.0540 | 20.00% |
| Cluster Popularity | 0.0427 | 0.1078 | 0.0564 | 48.33% |
| Hybrid Recommendation | 0.0470 | 0.1180 | 0.0619 | 98.33% |

The hybrid approach achieved the highest F1@5 among the tested methods.

These metrics compare recommendations with students' later historical course selections. They do not directly measure satisfaction or learning outcomes.

## Streamlit Application

The application allows users to:

- Select a student ID.
- View student learning profiles.
- View assigned learner segments.
- Generate personalized course recommendations.
- Download recommendations as CSV.

## Limitations

- Recommendations are evaluated using historical course selections as a proxy for interest.
- The absolute F1@5 score indicates room for improvement.
- Real-time feedback and course completion data are not available.
- The system has not been evaluated through a live user experiment.

## Future Improvements

- Incorporate student feedback.
- Explore collaborative filtering.
- Improve recommendation relevance.
- Add course filters and interactive visualizations.
- Conduct real-world user testing.

## Conclusion

EduPro demonstrates an end-to-end application of data analytics and machine learning for student segmentation and personalized course recommendations.
