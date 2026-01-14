const express = require("express");
const path = require("path");
const app = express();

// Отдаём статические файлы
app.use(express.static(path.join(__dirname)));

// Порт, который Railway использует
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server started on port ${PORT}`);
});
