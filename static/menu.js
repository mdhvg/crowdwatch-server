fetch('/clear', {
    method: 'GET'
}).then(r => r.json()).then(json => console.log(json))

const tiles = document.querySelectorAll('.tile.video');

tiles.forEach(tile => {
    tile.addEventListener('click', async () => {
        const image = tile.querySelector('img');
        let src = image.src;
        await fetch('/setMap', {
            method: 'POST',
            body: JSON.stringify({ tile: src.substring(src.lastIndexOf("/") + 1, src.indexOf(".")) }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
    })
});

const liveTile = document.querySelector('.tile.live');

liveTile.addEventListener('click', async () => {
    await fetch('/setMap', {
        method: 'POST',
        body: JSON.stringify({ tile: 'Live' }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
})