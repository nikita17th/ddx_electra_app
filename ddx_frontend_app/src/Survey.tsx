import React, {useEffect, useState, useMemo} from 'react';
import axios from 'axios';
import {Alert, Box, Button, Chip, CircularProgress, Container, LinearProgress, Paper, Slide} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import {Language, QuestionType} from "./types";
import QuestionComponent from "./components/QuestionComponent";
import {useNavigate} from "react-router-dom";

interface QuestionData {
    [key: string]: QuestionType;
}

const validateQuestionsData = (data: any): QuestionData => {
    const validatedData: QuestionData = {};
    Object.entries(data).forEach(([key, question]) => {
        const typedQuestion = question as QuestionType;
        if (
            ['M', 'C'].includes(typedQuestion.data_type) &&
            (!typedQuestion['possible-values'] || !Array.isArray(typedQuestion['possible-values']))
        ) {
            console.error(`Question ${key} has invalid possible-values`);
            return;
        }
        validatedData[key] = typedQuestion;
    });
    return validatedData;
};


const Survey: React.FC = () => {
    const navigate = useNavigate();
    const [language, setLanguage] = useState<Language>('ru');
    const [questionsData, setQuestionsData] = useState<QuestionData>({});
    const [isLoading, setIsLoading] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);
    useEffect(() => {
        const fetchQuestions = async () => {
            try {
                const response = await axios.get('/api/questions');
                const validatedData = validateQuestionsData(response.data.questions);
                setQuestionsData(validatedData);
                setIsLoading(false);
            } catch (err) {
                setLoadError('Failed to load questions');
                setIsLoading(false);
            }
        };

        fetchQuestions();
    }, []);


    const typedQuestionsData = useMemo(() => validateQuestionsData(questionsData), [questionsData]);
    const questionKeys = Object.keys(typedQuestionsData);

    const [answers, setAnswers] = useState<Record<string, any>>(() => {
        const saved = localStorage.getItem('surveyProgress');
        if (saved) return JSON.parse(saved).answers;

        const initialAnswers: Record<string, any> = {};
        Object.values(typedQuestionsData).forEach(question => {
            initialAnswers[question.name] =
                question.data_type === 'B' ? (question.default_value ?? false) :
                    question.data_type === 'M' ? (Array.isArray(question.default_value) ? question.default_value : []) :
                        question.default_value;
        });
        return initialAnswers;
    });

    const groupedQuestions = useMemo(() => {
        const groups: { [key: string]: QuestionType[] } = {};
        Object.values(typedQuestionsData).forEach(question => {
            const code = question.code_question;
            if (!groups[code]) {
                groups[code] = [];
            }
            groups[code].push(question);
        });
        return groups;
    }, [typedQuestionsData]);


    const [currentStep, setCurrentStep] = useState(() => {
        const saved = localStorage.getItem('surveyProgress');
        return saved ? JSON.parse(saved).step : 0;
    });


    const groupKeys = Object.keys(groupedQuestions);
    const currentGroup = groupKeys[currentStep];
    const currentQuestions = groupedQuestions[currentGroup] || [];


    const [submitStatus, setSubmitStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

    useEffect(() => {
        const progress = {
            step: currentStep,
            answers: answers
        };
        try {
            localStorage.removeItem('surveyProgress');
            localStorage.setItem('surveyProgress', JSON.stringify(progress));
        } catch (e) {
            console.error('Failed to save progress:', e);
        }
    }, [currentStep, answers]);

    const currentQuestion = typedQuestionsData[questionKeys[currentStep]];
    const handleAnswer = (questionId: string, value: any) => {
        setAnswers(prev => ({
            ...prev,
            [questionId]: value
        }));
    };

    const progressValue = ((currentStep + 1) / groupKeys.length) * 100;

    const updateStep = (direction: 'next' | 'prev') => {
        const updatedAnswers = {...answers};
        currentQuestions.forEach(question => {
            if (updatedAnswers[question.name] === undefined) {
                updatedAnswers[question.name] = question.default_value;
            }
        });
        setAnswers(updatedAnswers);
        setCurrentStep((prev: number) => direction === 'next' ? prev + 1 : prev - 1);
    };


    const handleSubmit = async () => {
        if (Object.keys(answers).length !== questionKeys.length) {
            setSubmitStatus('error');
            return;
        }

        let profileData: Record<string, any> = {};
        try {
            const localProfile = localStorage.getItem('profileData');
            profileData = localProfile
                ? JSON.parse(localProfile)
                : (await axios.get('/api/profile')).data.answers || {};
        } catch (error) {
            console.error('Ошибка загрузки профиля:', error);
        }

        const transformedData = {
            answers: {
                ...profileData,

                ...answers
            }
        };

        setSubmitStatus('loading');
        try {
            console.log(transformedData);
            const response = await axios.post('/api/predict', transformedData);
            // localStorage.removeItem('surveyProgress');
            setSubmitStatus('success');
            navigate('/results', { state: { predictions: response.data } });
        } catch (error) {
            setSubmitStatus('error');
        }
    };

    if (isLoading) {
        return (
            <Container maxWidth="md" sx={{py: 4}}>
                <CircularProgress/>
            </Container>
        );
    }

    if (loadError) {
        return (
            <Container maxWidth="md" sx={{py: 4}}>
                <Alert severity="error">{loadError}</Alert>
            </Container>
        );
    }
    if (!currentQuestion) {
        return <CircularProgress/>;
    }

    const isGroupValid = currentQuestions.every(question => {
        const answer = answers[question.name];
        const validatedAnswer = answer !== undefined ? answer : question.default_value;

        if (question.is_antecedent) {
            if (question.data_type === 'M') return validatedAnswer?.length > 0;
            return validatedAnswer !== null && validatedAnswer !== '';
        }
        return true;
    });
    return (
        <Container maxWidth="md" sx={{py: 4}}>
            <Box sx={{mb: 4, display: 'flex', gap: 1, justifyContent: 'flex-end'}}>
                {['en', 'fr', 'ru'].map((lang) => (
                    <Chip
                        key={lang}
                        label={lang.toUpperCase()}
                        onClick={() => setLanguage(lang as Language)}
                        color={language === lang ? 'primary' : 'default'}
                        variant={language === lang ? 'filled' : 'outlined'}
                    />
                ))}
            </Box>

            <LinearProgress
                variant="determinate"
                value={progressValue}
                sx={{mb: 4, height: 8, borderRadius: 4}}
            />

            {submitStatus === 'success' && (
                <Alert severity="success" sx={{mb: 3}}>
                    {language === 'en' ? 'Submitted successfully!' :
                        language === 'fr' ? 'Envoyé avec succès!' :
                            'Данные успешно отправлены!'}
                </Alert>
            )}

            {submitStatus === 'error' && (
                <Alert severity="error" sx={{mb: 3}}>
                    {language === 'en' ? 'Please answer all questions' :
                        language === 'fr' ? 'Veuillez répondre à toutes les questions' :
                            'Пожалуйста, ответьте на все вопросы'}
                </Alert>
            )}

            <Slide in direction="up" timeout={300}>
                <Paper elevation={3} sx={{p: 3, mb: 3, borderRadius: 4}}>
                    {currentQuestions.map(question => (
                        <QuestionComponent
                            key={question.name}
                            currentQuestion={question}
                            language={language}
                            answers={answers}
                            handleAnswer={handleAnswer}
                        />
                    ))}
                </Paper>
            </Slide>

            <Box sx={{display: 'flex', justifyContent: 'space-between'}}>
                <Button
                    onClick={() => updateStep('prev')}
                    disabled={currentStep === 0}
                    variant="outlined"
                >
                    {language === 'en' ? 'Back' : language === 'fr' ? 'Retour' : 'Назад'}
                </Button>

                {currentStep === groupKeys.length - 1 ? (
                    <Button
                        variant="contained"
                        onClick={handleSubmit}
                        disabled={submitStatus === 'loading' || !isGroupValid}
                        endIcon={submitStatus === 'loading' ? <CircularProgress size={24}/> : <SendIcon/>}
                    >
                        {language === 'en' ? 'Submit' : language === 'fr' ? 'Soumettre' : 'Отправить'}
                    </Button>
                ) : (
                    <Button
                        variant="contained"
                        onClick={() => updateStep('next')}
                        disabled={!isGroupValid}
                        sx={{'&:disabled': {backgroundColor: 'action.disabledBackground'}}}
                    >
                        {language === 'en' ? 'Next' : language === 'fr' ? 'Suivant' : 'Далее'}
                    </Button>
                )}
            </Box>
        </Container>
    );
};

export default Survey;