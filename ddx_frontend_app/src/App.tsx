import { Routes, Route } from 'react-router-dom';
import HomePage from './HomePage';
import Profile from './Profile';
import Survey from './Survey';
import ProtectedRoute from './ProtectedRoute';
import ResultsPage from "./ResultPage";

const App: React.FC = () => {
    return (
        <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/profile" element={<Profile />} />
            <Route
                path="/survey"
                element={
                    <ProtectedRoute>
                        <Survey />
                    </ProtectedRoute>
                }
            />
            <Route path="/results" element={<ResultsPage />} />
        </Routes>
    );
};

export default App;