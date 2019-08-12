function parseFile(name) {
    return new Promise(function (complete, error) {
        Papa.parse(name, {
            delimiter: ',',
            download: true,
            header: true,
            complete: complete,
        });
    });
}

function drawImageAndRegion(canvas, src, ann) {
    let img = new Image();
    img.onload = function () {
        let orig_h = img.height;
        let orig_w = img.width;
        let ratio = orig_w / orig_h;
        let w = 300;
        let h = w / ratio;
        let scale = orig_w / w;
        canvas.width = w;
        canvas.height = h;
        let ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(img, 0, 0, w, h);
        ctx.lineWidth = "4";
        ctx.strokeStyle = "#f6416c";
        let [box_xo, box_yo, box_wo, box_ho] = ann.bbox;
        let box_x = box_xo / scale;
        let box_y = box_yo / scale;
        let box_w = box_wo / scale;
        let box_h = box_ho / scale;
        ctx.rect(box_x, box_y, box_w, box_h);
        ctx.stroke()
    }
    img.src = src;
}

function getAnns(obj, blacklist, lim = 3) {
    let found = [];
    for (let key in obj) {
        if (!obj.hasOwnProperty(key)) {
            continue;
        }
        let skip = false;
        for (let b of blacklist) {
            if (key == b) {
                skip = true;
                break;
            }
        }
        if (skip) {
            continue;
        }
        val = obj[key];
        if (val == "1") {
            found.push(key);
            if (found.length == lim) {
                break;
            }
        }
    }
    return found;
}

function getAffs(vps, want_lim = 3) {
    // we want to return lim unique items, sampled at the frequency they appear in vps.
    // however, we set a lower limit if there are fewer than lim unique items in the
    // list itself.
    let lim = Math.min(_.uniq(vps).length, want_lim);
    let picked = [];
    while (picked.length < lim) {
        picked.push(_.sample(vps));
        picked = _.uniq(picked);
    }
    return picked;
}

async function parseData() {
    let coco = await $.getJSON('data/coco.json');
    let situatedProps = await parseFile('data/situated-properties.csv');
    // let situatedAffs = await parseFile('data/situated-affordances.csv');
    let situatedAffs = await parseFile('data/situated-affordances-full.csv');

    let prop_idx = Math.floor(Math.random() * situatedProps.data.length);
    let prop = situatedProps.data[prop_idx];
    let imgID = prop['cocoImgID'];
    let img = coco.images.find(function (el) {
        return el.id == imgID;
    });
    let annID = prop['cocoAnnID'];
    let ann = coco.annotations.find(function (el) {
        return el.id == annID && el.image_id == imgID;
    });
    let aff = situatedAffs.data.find(function (el) {
        return el.cocoImgID == imgID && el.cocoAnnID == annID;
    });

    drawImageAndRegion(document.getElementById('demoImg'), img.coco_url, ann)

    let foundProps = getAnns(prop, ['cocoImgID', 'cocoAnnID', 'objectUID']);

    let buf = '';
    if (foundProps.length > 0) {
        buf += '<br />How would you describe the <span class="pa1" style="color: white; background-color: #f6416c;">' + aff.objectHuman + '</span> ?<ul class="examples">\n';
        for (let f of foundProps) {
            buf += '<li><span class="code">' + f + '</span> ?</li>\n';
        }
        buf += '</ul>'
    }

    if (aff.vps.length > 0) {
        buf += '<br />What might you do to the <span class="pa1" style="color: white; background-color: #f6416c;">' + aff.objectHuman + '</span> ?<ul class="examples">\n';
        let all_vps = aff.vps.split(';')
        let vps = getAffs(all_vps);
        let obj_ref = aff.objectHuman == 'person' ? 'them' : 'it';
        for (let vp of vps) {
            buf += '<li>' + vp + ' ' + obj_ref + '?</li>\n';
        }
        buf += '</ul>'
    }

    document.getElementById('propContent').innerHTML = buf
}

// main
window.onload = parseData;
