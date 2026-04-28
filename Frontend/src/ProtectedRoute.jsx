import { Navigate } from "react-router-dom";

export default function ProtectedRoute({ children, requiredRole }) {
  const token = localStorage.getItem("token");
  const role = localStorage.getItem("role");

  if (!token) {
    return <Navigate to="/login" />;
  }

  // If role restriction is applied
  if (requiredRole && role !== requiredRole) {
    return <Navigate to="/app" />; // or unauthorized page
  }

  return children;
}
