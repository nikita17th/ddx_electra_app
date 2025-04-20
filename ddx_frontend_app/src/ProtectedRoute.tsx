import React, {useEffect, useState} from 'react';
import {Navigate} from 'react-router-dom';
import axios from 'axios';
import {CircularProgress} from "@mui/material";

const ProtectedRoute: React.FC<{ children: React.ReactElement }> = ({children}) => {
    const [isVerified, setIsVerified] = useState<boolean | null>(null);

    useEffect(() => {
        const verifyProfile = async () => {
            try {
                const response = await axios.get('/api/profile/status');
                setIsVerified(response.data.has_profile);
            } catch (error) {
                console.error('Profile verification failed:', error);
                setIsVerified(false);
            }
        };
        verifyProfile();
    }, []);

    if (isVerified === null) {
        return <CircularProgress sx={{display: 'block', mt: 4, mx: 'auto'}}/>;
    }

    return isVerified ? children : <Navigate to="/" replace/>;
};

export default ProtectedRoute;