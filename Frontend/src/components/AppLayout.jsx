import { Outlet } from "react-router-dom";
import AppHeader from "./AppHeader.jsx";
import AppFooter from "./AppFooter.jsx";
import styles from "./AppLayout.module.css";

export default function AppLayout() {
  return (
    <div className={styles.shell}>
      <AppHeader />
      <main className={styles.main}>
        <div className={styles.workArea}>
          <Outlet />
        </div>
      </main>
      <AppFooter />
    </div>
  );
}
