import React, { useEffect, useState } from 'react';
import { Button, Container, Typography, Alert, CircularProgress } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const HomePage: React.FC = () => {
    const [hasProfile, setHasProfile] = useState<boolean | null>(null);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    useEffect(() => {
        const checkProfile = async () => {
            try {
                const response = await axios.get('/api/profile/status');
                setHasProfile(response.data.has_profile);
                setLoading(false);
            } catch (error) {
                console.error('Error checking profile:', error);
                setLoading(false);
            }
        };
        checkProfile();
    }, []);

    if (loading) {
        return <CircularProgress sx={{ display: 'block', mt: 4, mx: 'auto' }} />;
    }

    return (
        <Container maxWidth="md" sx={{ textAlign: 'center', mt: 8 }}>
            <Typography variant="h3" gutterBottom>
                Медицинская диагностическая система
            </Typography>

            <div style={{ marginTop: '3rem' }}>
                <Button
                    variant="contained"
                    size="large"
                    sx={{ mr: 2 }}
                    onClick={() => navigate('/profile')}
                >
                    Редактировать профиль
                </Button>

                <Button
                    variant="contained"
                    size="large"
                    disabled={!hasProfile}
                    onClick={() => navigate('/survey')}
                >
                    Пройти опрос
                </Button>
            </div>

            {!hasProfile && (
                <Alert severity="warning" sx={{ mt: 4 }}>
                    Для прохождения опроса необходимо сначала заполнить профиль!
                </Alert>
            )}
        </Container>
    );
};

export default HomePage;