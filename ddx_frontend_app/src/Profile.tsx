import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
    Alert, Box,
    Button,
    Chip,
    CircularProgress,
    Container,
    Grid,
    Paper,
    Typography
} from '@mui/material';
import QuestionComponent from "./components/QuestionComponent";
import { Language, QuestionType } from './types';
import { useNavigate } from 'react-router-dom';

const Profile: React.FC = () => {
    const [questions, setQuestions] = useState<QuestionType[]>([]);
    const [answers, setAnswers] = useState<Record<string, any>>(() => {
        const saved = localStorage.getItem('handleSave');
        return saved ? JSON.parse(saved) : {};
    });
    const [isLoading, setIsLoading] = useState(true);
    const [saveStatus, setSaveStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
    const [language, setLanguage] = useState<Language>('ru');
    const navigate = useNavigate();

    useEffect(() => {
        const loadData = async () => {
            try {
                const [questionsRes, profileRes] = await Promise.all([
                    axios.get('/api/profile/questions'),
                    axios.get('/api/profile')
                ]);

                const mergedAnswers = questionsRes.data.questions.reduce((acc: any, q: QuestionType) => ({
                    ...acc,
                    [q.name]: q.data_type === 'B'
                        ? false
                        : profileRes.data.answers[q.name] ?? q.default_value
                }), {});

                setQuestions(questionsRes.data.questions);
                setAnswers(mergedAnswers);
                setIsLoading(false);
            } catch (error) {
                console.error('Error loading profile:', error);
                setIsLoading(false);
            }
        };
        loadData();
    }, []);

    const handleAnswer = (questionId: string, value: any) => {
        setAnswers(prev => {
            const newAnswers = {...prev, [questionId]: value};
            localStorage.setItem('profileDraft', JSON.stringify(newAnswers));
            return newAnswers;
        });
    };

    const handleSave = async () => {
        setSaveStatus('loading');
        try {
            await axios.put('/api/profile', answers);
            localStorage.setItem('profileData', JSON.stringify(answers));
            setSaveStatus('success');
            setTimeout(() => setSaveStatus('idle'), 2000);
            navigate('/');
        } catch (error) {
            setSaveStatus('error');
        }
    };

    if (isLoading) return <CircularProgress sx={{ display: 'block', mt: 4, mx: 'auto' }} />;

    return (
        <Container maxWidth="md" sx={{ py: 4 }}>
            <Box sx={{ mb: 4, display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
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
            <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h4" gutterBottom>
                    {language === 'en' ? 'Profile' :
                        language === 'fr' ? 'Profil' : 'Профиль'}
                </Typography>

                <Grid container spacing={3}>
                    {questions.map(question => (
                        <Grid item xs={12} md={6} key={question.name}>
                            <QuestionComponent
                                currentQuestion={question}
                                language={language}
                                answers={answers}
                                handleAnswer={handleAnswer}
                            />
                        </Grid>
                    ))}
                </Grid>

                <Button
                    variant="contained"
                    size="large"
                    onClick={handleSave}
                    disabled={saveStatus === 'loading'}
                    sx={{ mt: 3, float: 'right' }}
                >
                    {saveStatus === 'loading' ? (
                        <CircularProgress size={24} />
                    ) : (
                        language === 'en' ? 'Save' :
                            language === 'fr' ? 'Sauvegarder' : 'Сохранить'
                    )}
                </Button>

                {saveStatus === 'success' && (
                    <Alert severity="success" sx={{ mt: 2 }}>
                        {language === 'en' ? 'Saved successfully!' :
                            language === 'fr' ? 'Sauvegardé avec succès!' :
                                'Успешно сохранено!'}
                    </Alert>
                )}
            </Paper>
        </Container>
    );
};

export default Profile;