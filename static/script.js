function addZero(i) {
    if (i < 10) {i = "0" + i}
    return i;
}
function startTime() {
    const d = new Date();
    const months = ["January","February","March","April","May","June","July","August","September","October","November","December"];
    const days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];
    let hours = addZero(d.getHours());
    let minutes = addZero(d.getMinutes());
    let date = d.getDate();
    let day = days[d.getDay()];
    let month = months[d.getMonth()];
    document.getElementById('clock').innerHTML =  hours + ":" + minutes;
    document.getElementById('date').innerHTML =  day + ", " + date + " " + month;
}
window.addEventListener("DOMContentLoaded", () => {
    startTime()
})